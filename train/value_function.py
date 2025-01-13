import os
import logging
import torch
import shutil
import tempfile

from huggingface_hub import Repository, HfApi
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
from accelerate import Accelerator
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train_utils import empty_cache_decorator
from typing import Optional
import os
from debug import set_trace

class AttrDict(dict):
    def __getattr__(self, attr):
        return self[attr]
    def __setattr__(self, attr, value):
        self[attr] = value

# Extract the hidden state of the last token of each response        
def get_embedding(hidden_states, attention_mask):
    # use third-from-last instead of last since last layer has thrown away the information we want
    all_hiddens = hidden_states[-3] # (2 * local_microbatch_size, sequence_length, hidden_size)

    # extract the hidden vector at the timestep of the eos token, i.e., the last token of each response
    # can use the attention mask to do this, since we know the eos token is the last token of each response
    # (2 * local_microbatch_size, sequence_length)
    eos_index = attention_mask.sum(-1) - 1

    # prepare the eos_index to do a gather on all_hiddens
    eos_index = eos_index.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, all_hiddens.shape[-1]) # (2 * local_microbatch_size, sequence_length, hidden_size)
    gathered_hiddens = all_hiddens.gather(1, eos_index).squeeze(1) # (2 * local_microbatch_size, hidden_size)

    return gathered_hiddens

# Extract the hidden state of the last token of each response        
def get_embedding_token(hidden_states, attention_mask):
    # use third-from-last instead of last since last layer has thrown away the information we want
    gathered_hiddens = hidden_states[-3] # (2 * local_microbatch_size, sequence_length, hidden_size)
    
    return gathered_hiddens

# Extract the hidden state of the last token of each response        
def get_embedding_mean(hidden_states, attention_mask):
    # use third-from-last instead of last since last layer has thrown away the information we want
    all_hiddens = hidden_states[-3] # (2 * local_microbatch_size, sequence_length, hidden_size)

    gathered_hiddens = all_hiddens.mean(dim=1, keepdim=False) # (2 * local_microbatch_size, hidden_size)

    return gathered_hiddens


def upload_and_clean_checkpoint(local_dir, model_id, model, tokenizer):
    """
    Upload a checkpoint to Hugging Face and remove the local copy after upload.
    """
    model.push_to_hub(model_id, private=True)
    tokenizer.push_to_hub(model_id, private=True)
    model.config.push_to_hub(model_id, private=True)
    print(f"Checkpoint uploaded to {model_id}. Removing local files...")
    shutil.rmtree(local_dir)
    print(f"Local checkpoint directory '{local_dir}' removed.")


class ValueFunction(nn.Module):
    def __init__(
        self,
        model_name: str = 'mistralai/Mistral-7B-Instruct-v0.2',
        model_checkpoint: Optional[str] = None,
        embedding_type: str = 'eos',
        torch_dtype: torch.dtype = torch.bfloat16,
        low_cpu_mem_usage: bool = True,
        device_map: str = 'auto',
        flash_attn: bool = False,
        activation_checkpointing: bool = True,
        discretize: bool = False,
        num_bins: int = 51,
        dropout: float = 0.1,
        cache_dir: str = '/home/anikait.singh/.cache/',
        token: str = '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',
        trust_remote_code: bool = True,
        accelerator: Optional[Accelerator] = None,
        special_tokens: Optional[dict] = None,
    ):
        super().__init__()
        self.model_name = model_name
        self.model_checkpoint = model_checkpoint
        self.embedding_type = embedding_type
        self.torch_dtype = torch_dtype
        self.low_cpu_mem_usage = low_cpu_mem_usage
        self.device_map = device_map
        self.flash_attn = flash_attn
        self.activation_checkpointing = activation_checkpointing
        self.discretize = discretize
        self.num_bins = num_bins
        self.dropout = dropout
        self.cache_dir = cache_dir        
        self.accelerator = accelerator
        
        shared_kwargs = dict(
            cache_dir=self.cache_dir,         
            trust_remote_code=trust_remote_code,
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            # pad_token_id=0,
            **shared_kwargs,
        )
        if special_tokens is not None:
            self.tokenizer.add_special_tokens(special_tokens)
        
        if self.accelerator is not None:
            self.accelerator.print(f"Tokenizer vocab size: {len(self.tokenizer)}")
            self.accelerator.print('EOS token:', self.tokenizer.eos_token)
            self.accelerator.print('PAD token:', self.tokenizer.pad_token)
            self.accelerator.print('Special tokens:', self.tokenizer.special_tokens_map)
        else:
            print(f"Tokenizer vocab size: {len(self.tokenizer)}")
            print('EOS token:', self.tokenizer.eos_token)
            print('PAD token:', self.tokenizer.pad_token)
            print('Special tokens:', self.tokenizer.special_tokens_map)
        
        device_args = dict(
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=self.low_cpu_mem_usage,
        )
        if self.device_map is not None:
            device_args.update(dict(device_map=self.device_map))

        if self.flash_attn:
            device_args.update(dict(attn_implementation="flash_attention_2"))

        no_checkpoint = (model_checkpoint is None or model_checkpoint == '' or model_checkpoint == 'None')
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name if no_checkpoint else model_checkpoint, 
            **device_args,
            **shared_kwargs,
        )

        # set the pad token to eos token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        # self.model.config.pad_token_id = self.model.config.eos_token_id
        
        if special_tokens is not None:
            self.model.resize_token_embeddings(len(self.tokenizer))
        
        if self.activation_checkpointing:
            self.model.config.use_cache = False
            self.model.gradient_checkpointing_enable()

        if self.discretize:
            self.project = nn.Linear(self.model.config.hidden_size, self.num_bins)
        else:
            self.project = nn.Linear(self.model.config.hidden_size, 1)
        self.project = self.project.to(self.torch_dtype)

        if self.accelerator is not None:
            self.model, self.project = self.accelerator.prepare(self.model, self.project)
            self.accelerator.register_for_checkpointing(self.model)
            self.accelerator.register_for_checkpointing(self.project)
            self.device = self.accelerator.device
        else:
            self.device = self.model.device
            self.project = self.project.to(self.device)


    @empty_cache_decorator
    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        
        if self.embedding_type == 'eos':
            emb = get_embedding(hidden_states, attention_mask)
        elif self.embedding_type == 'mean':
            emb = get_embedding_mean(hidden_states, attention_mask)
        else:
            raise ValueError(f"Unknown embedding type {self.embedding_type}")

        emb = F.dropout(emb, p=self.dropout, training=self.training)
        emb = emb.to(self.device).to(self.torch_dtype)

        logits = self.project(emb)
        del outputs, hidden_states, emb
        return AttrDict(logits=logits)
    

    @empty_cache_decorator
    def save(self, path: str):
        self.accelerator.wait_for_everyone()
        aprint = self.accelerator.print if self.accelerator is not None else print

        model_path, project_path, model_transformers = f'{path}/model.pth', f'{path}/project.pth', f'{path}/model_transformers'
        os.makedirs(model_transformers, exist_ok=True)

        unwrapped_model = self.accelerator.unwrap_model(self.model)

        aprint(f"Value Function: Saving project to {project_path}")
        self.accelerator.save(self.accelerator.get_state_dict(self.project), project_path)
        
        aprint(f"Value Function: Saving model to {model_path}")
        self.accelerator.save(self.accelerator.get_state_dict(self.model), model_path)
        
        aprint(f"Value Function: Saving model w/Transformers to {model_path}")
        unwrapped_model.save_pretrained(
            model_transformers,
            save_function=self.accelerator.save, 
            state_dict=self.accelerator.get_state_dict(self.model),
        )        


    @empty_cache_decorator
    def load(self, path: str):
        model_path, project_path = f'{path}/model.pth', f'{path}/project.pth'
        
        assert os.path.exists(model_path), f"Model path {model_path} does not exist"
        assert os.path.exists(project_path), f"Project path {project_path} does not exist"
        
        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            unwrapped_project = self.accelerator.unwrap_model(self.project)
        else:
            unwrapped_model = self.model
            unwrapped_project = self.project
        
        loaded_state_dict = torch.load(project_path, map_location='cpu')
        unwrapped_project.load_state_dict(loaded_state_dict)
        
        loaded_state_dict = torch.load(model_path, map_location='cpu')
        unwrapped_model.load_state_dict(loaded_state_dict)
        

class ParallelValueFunction(nn.Module):
    def __init__(
        self,
        model_name: str = 'mistralai/Mistral-7B-Instruct-v0.2',
        model_checkpoint: Optional[str] = None,
        embedding_type: str = 'token',
        torch_dtype: torch.dtype = torch.bfloat16,
        low_cpu_mem_usage: bool = True,
        device_map: str = 'auto',
        flash_attn: bool = False,
        activation_checkpointing: bool = True,
        discretize: bool = False,
        num_bins: int = 51,
        dropout: float = 0.1,
        cache_dir: str = '',
        trust_remote_code: bool = True,
        accelerator: Optional[Accelerator] = None,
        special_tokens: Optional[dict] = None,
    ):
        super().__init__()
        self.model_name = model_name
        self.model_checkpoint = model_checkpoint
        self.embedding_type = embedding_type
        self.torch_dtype = torch_dtype
        self.low_cpu_mem_usage = low_cpu_mem_usage
        self.device_map = device_map
        self.flash_attn = flash_attn
        self.activation_checkpointing = activation_checkpointing
        self.discretize = discretize
        self.num_bins = num_bins
        self.dropout = dropout
        self.cache_dir = cache_dir
        self.accelerator = accelerator        
        
        shared_kwargs = dict(
            cache_dir=self.cache_dir,            
            trust_remote_code=trust_remote_code,
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            **shared_kwargs,
        )
        if special_tokens is not None:
            self.tokenizer.add_special_tokens(special_tokens)
        
        if self.accelerator is not None:
            self.accelerator.print(f"Tokenizer vocab size: {len(self.tokenizer)}")
            self.accelerator.print('EOS token:', self.tokenizer.eos_token)
            self.accelerator.print('PAD token:', self.tokenizer.pad_token)
            self.accelerator.print('Special tokens:', self.tokenizer.special_tokens_map)
        else:
            print(f"Tokenizer vocab size: {len(self.tokenizer)}")
            print('EOS token:', self.tokenizer.eos_token)
            print('PAD token:', self.tokenizer.pad_token)
            print('Special tokens:', self.tokenizer.special_tokens_map)
        
        device_args = dict(
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=self.low_cpu_mem_usage,
        )
        if self.device_map is not None:
            device_args.update(dict(device_map=self.device_map))

        if self.flash_attn:
            device_args.update(dict(attn_implementation="flash_attention_2"))
        
        no_checkpoint = (model_checkpoint is None or model_checkpoint == '' or model_checkpoint == 'None')
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name if no_checkpoint else model_checkpoint, 
            **device_args,
            **shared_kwargs,
        )        

        # set the pad token to eos token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        # self.model.config.pad_token_id = self.model.config.eos_token_id
        
        if special_tokens is not None:
            self.model.resize_token_embeddings(len(self.tokenizer))
        
        if self.activation_checkpointing:
            self.model.config.use_cache = False
            self.model.gradient_checkpointing_enable()

        if self.discretize:
            self.project = nn.Linear(self.model.config.hidden_size, self.num_bins)
        else:
            self.project = nn.Linear(self.model.config.hidden_size, 1)
        self.project = self.project.to(self.torch_dtype)

        if self.accelerator is not None:
            self.model, self.project = self.accelerator.prepare(self.model, self.project)
            self.accelerator.register_for_checkpointing(self.model)
            self.accelerator.register_for_checkpointing(self.project)
            self.device = self.accelerator.device
        else:
            self.device = self.model.device
            self.project = self.project.to(self.device)

    @empty_cache_decorator
    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        
        if self.embedding_type == 'token':
            emb = get_embedding_token(hidden_states, attention_mask)
        else:
            raise ValueError(f"Unknown embedding type {self.embedding_type}")

        emb = F.dropout(emb, p=self.dropout, training=self.training)
        emb = emb.to(self.device).to(self.torch_dtype)

        # (2 * local_microbatch_size, sequence_length, hidden_size) -> (2 * local_microbatch_size, num_bins/1)
        logits = self.project(emb)
        del outputs, hidden_states, emb
        return AttrDict(logits=logits)
    
    @empty_cache_decorator
    def save(self, path: str, model_id: str = None):
        self.accelerator.wait_for_everyone()
        aprint = self.accelerator.print if self.accelerator is not None else print

        model_path, project_path, model_transformers = f'{path}/model.pth', f'{path}/project.pth', f'{path}/model_transformers'
        os.makedirs(model_transformers, exist_ok=True)

        unwrapped_model = self.accelerator.unwrap_model(self.model)

        aprint(f"Value Function: Saving project to {project_path}")
        self.accelerator.save(self.accelerator.get_state_dict(self.project), project_path)
        
        aprint(f"Value Function: Saving model to {model_path}")
        self.accelerator.save(self.accelerator.get_state_dict(self.model), model_path)
        
        aprint(f"Value Function: Saving model w/Transformers to {model_path}")
        unwrapped_model.save_pretrained(
            model_transformers,
            save_function=self.accelerator.save, 
            state_dict=self.accelerator.get_state_dict(self.model),
        )

        if model_id:
            upload_and_clean_checkpoint(path, model_id, unwrapped_model, self.tokenizer)


    @empty_cache_decorator
    def load(self, path: str):
        model_path, project_path = f'{path}/model.pth', f'{path}/project.pth'
        
        assert os.path.exists(model_path), f"Model path {model_path} does not exist"
        assert os.path.exists(project_path), f"Project path {project_path} does not exist"
        
        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            unwrapped_project = self.accelerator.unwrap_model(self.project)
        else:
            unwrapped_model = self.model
            unwrapped_project = self.project
        
        loaded_state_dict = torch.load(project_path, map_location='cpu')
        unwrapped_project.load_state_dict(loaded_state_dict)
        
        loaded_state_dict = torch.load(model_path, map_location='cpu')
        unwrapped_model.load_state_dict(loaded_state_dict)

if __name__ == '__main__':
    things_to_encode = ['hello', 'goodbye', 'what is the meaning of life']

    for discretize in [True, False]:
        for emb_type in ['eos', 'mean']:
            print(f"Testing discretize={discretize}, emb_type={emb_type}")
            output_shape = (3, 51) if discretize else (3, 1)
    
            model = ValueFunction(embedding_type=emb_type, discretize=discretize, model_name='EleutherAI/pythia-70m')
            tok = model.tokenizer(things_to_encode, return_tensors='pt', padding=True, truncation=True, max_length=128)
            input_ids, attention_mask = tok['input_ids'], tok['attention_mask']
            output = model(input_ids, attention_mask)
            assert output.logits.shape == output_shape, f"Expected shape {output_shape} but got {output.logits.shape}"
            del model, tok, input_ids, attention_mask, output
 
        model = ParallelValueFunction(embedding_type='token', model_name='EleutherAI/pythia-70m')
        tok = model.tokenizer(things_to_encode, return_tensors='pt', padding=True, truncation=True, max_length=128)
        sequence_length = tok['input_ids'].shape[1]
        input_ids, attention_mask = tok['input_ids'], tok['attention_mask']
        output = model(input_ids, attention_mask)
        output_shape = ((3, sequence_length, 51) if discretize else (3, sequence_length, 1))
        assert output.logits.shape == output_shape, f"Expected shape {output_shape} but got {output.logits.shape}"
 
    print('All tests passed!')