from typing import Dict, Sequence, List
from dataclasses import dataclass
import torch
import copy
import transformers 
from datasets import Dataset
from omegaconf import DictConfig

# from stargate.vllm_inference_model import VLLMInferenceModel

IGNORE_INDEX = -100


def get_formatted_responses(
    model: any, #VLLMInferenceModel, 
    tokenizer: transformers.AutoTokenizer,
    prompts: List[str], 
    config: DictConfig,
    output_format: str="Clarifying Question:",
    invalid_output: str="<|invalid_response|>",
) -> List[str]:
    """Formats prompts and returns formatted model responses."""
    formatted_prompts = [
        tokenizer.apply_chat_template(prompt, tokenize=False) 
        for prompt in prompts
    ]
    
    if output_format == "Roleplayer":
        formatted_prompts = [prompt[:-10] for prompt in formatted_prompts]
    
    # breakpoint()
    
    responses = model.batch_prompt(
        prompts=formatted_prompts,
        **config,
    )
    
    formatted_responses = []

    for i, response in enumerate(responses):
        try:
            if output_format == "Roleplayer":
                formatted_responses.append(f"{response.strip()}")
            else:
                formatted_responses.append(response.split(output_format)[1].strip())
        except:
            formatted_responses.append(invalid_output)
            
    return formatted_responses

    
def mutual_information(
    logprobs: torch.FloatTensor, 
    n_users: int,
) -> torch.FloatTensor:
    """Computes mutual information."""
    # uniform over users 
    p_user = torch.tensor(1/n_users).repeat(n_users)
    
    # conditional probs 
    p_response_given_user = ((logprobs - torch.logsumexp(logprobs, dim=0))).exp()
    
    # marginal probs 
    p_response = (p_response_given_user * p_user).sum()
    
    # joint 
    p_response_and_user = p_response_given_user * p_user 
    
    # mutual information
    mutual_information = p_response_and_user * torch.log(p_response_given_user / p_response)
    
    return mutual_information.sum()

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Data collator for SFT which masks user from the loss."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(
        self, 
        instances: Sequence[Dict],
    ) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        
        
def _tokenize_fn(
    messages: Sequence[Dict], 
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:  
    """Tokenize list of chat messages (user, assistant). TODO: Add support for system message."""
    inputs, labels, attention_masks = [], [], []
    step_token = tokenizer.encode("\n\n")[1]
    user_message = messages[0]
    assistant_message = messages[1]
    
    user_tokenized = tokenizer.apply_chat_template(
        [user_message],
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
    )[0]
    assistant_tokenized = tokenizer.apply_chat_template(
        [assistant_message],
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
    )[0][1:]    # skip bos_token    
            
    n_steps = len(assistant_message['content'].strip().split("\n\n")) - 1
    for i in range(n_steps):
        step_inputs, step_labels = [], []         
        step_inputs.append(user_tokenized)        
        masked_labels = torch.full(user_tokenized.shape, IGNORE_INDEX, dtype=torch.long)
        step_labels.append(masked_labels)
        # step_inds = torch.where(user_tokenized == step_token)[0][1:]    # skip the first "\n\n" added for user header
       
        # assistant response we need to mask scores for future steps
        label = copy.deepcopy(assistant_tokenized)        
        step_inds = torch.where(assistant_tokenized == step_token)[0][1:]
        step_after = step_inds[i].item() + 1        
        # mask 
        label[step_after:] = IGNORE_INDEX
        step_inputs.append(assistant_tokenized)
        step_labels.append(label)
        step_inputs = torch.cat(step_inputs, dim=0)
        step_labels = torch.cat(step_labels, dim=0)
        step_attentino_masks = torch.ones_like(step_inputs)

        inputs.append(step_inputs)
        labels.append(step_labels)
        attention_masks.append(step_attentino_masks)        
            
    # input_ids = torch.cat(inputs, dim=0)
    # labels = torch.cat(labels, dim=0)   
    input_ids = torch.stack(inputs, dim=0)
    labels = torch.stack(labels, dim=0)
    attention_masks = torch.stack(attention_masks, dim=0) 
    
    return dict(
        input_ids=input_ids,
        labels=labels,
        attention_mask=attention_masks
    )
    
    
def preprocess(
    targets: List[str],
    tokenizer: transformers.PreTrainedTokenizer,
):
    """Preprocess the data by tokenizing."""
    tokenized = {}
    i = 0 
    for messages in targets:
        # tokenized[i] = _tokenize_fn(messages, tokenizer)
        tokenized_dict = _tokenize_fn(messages, tokenizer)
        for input_ids, labels, attention_mask in \
            zip(tokenized_dict["input_ids"], tokenized_dict["labels"], tokenized_dict["attention_mask"]):
            tokenized[i] = dict(
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask
            )
            i += 1

    tokenized_formatted = dict(
        input_ids=[example["input_ids"] for example in tokenized.values()],
        labels=[example["labels"] for example in tokenized.values()],
        attention_mask=[example["attention_mask"] for example in tokenized.values()]
    )
    dataset = Dataset.from_dict(tokenized_formatted)
    dataset.set_format('torch')

    return dataset