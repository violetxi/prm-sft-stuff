import os
import tqdm
import torch
import numpy as np
import random
import hydra
import logging
import shutil


from omegaconf import DictConfig, OmegaConf
from datasets import load_dataset
from collections import defaultdict
from accelerate import Accelerator
from accelerate.utils import GradientAccumulationPlugin
from value_function import ValueFunction
from transformers import get_cosine_schedule_with_warmup
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers.tokenization_utils_base import BatchEncoding
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
warnings.filterwarnings("ignore")
import transformers
import datasets
transformers.logging.set_verbosity_error()
from debug import set_trace
from train_utils import empty_cache_decorator



def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    return seed


def get_dataset(dataset_name, tokenizer=None):
    if dataset_name == 'numina-tir':
        ds = datasets.load_dataset('violetxi/NuminaMath-TIR-Filtered')
    else:
        raise ValueError(f"Dataset {dataset_name} not found")
    return ds


@empty_cache_decorator
def save_model(accelerator, model, tokenizer, path, model_id=None):    
    # Create all parent directories    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model_path = f'{path}/model.pth'
    transformers_path = f'{path}/model_transformers'
    
    accelerator.wait_for_everyone()

    os.makedirs(transformers_path, exist_ok=True)
    
    unwrapped_model = accelerator.unwrap_model(model)

    accelerator.print(f"Saving model to {model_path}")
    accelerator.save(accelerator.get_state_dict(model.model), model_path)
        
    accelerator.print(f"Saving model w/Transformers to {transformers_path}")
    unwrapped_model.save_pretrained(transformers_path, save_function=accelerator.save, state_dict=accelerator.get_state_dict(model.model))
    
    if model_id:
        model.push_to_hub(model_id, private=True)
        tokenizer.push_to_hub(model_id, private=True)
        model.config.push_to_hub(model_id, private=True)
        accelerator.print(f"Uploaded model to {model_id}")
        shutil.rmtree(path)
        accelerator.print(f"Removed local checkpoint directory '{path}'")


@hydra.main(version_base=None, config_path="config", config_name="sft_train")
def main(args: DictConfig):
    logging.info(f"Running with config: {args}")
    set_seed(args.seed)
    if args.gradient_accumulation_steps > 1:
        plugin = GradientAccumulationPlugin(
            num_steps=args.gradient_accumulation_steps,            
            adjust_scheduler=False,
            sync_with_dataloader=False,
            sync_each_batch=True,
        )
    else:
        plugin = None
    
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with="wandb",
        gradient_accumulation_plugin=plugin,
    )

    ds = get_dataset(args.dataset_name)    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_checkpoint if args.model_checkpoint else args.model_name,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=args.low_cpu_mem_usage,
        cache_dir=args.cache_dir,
        trust_remote_code=True,
        device_map="auto"
    )
    model = accelerator.prepare(model)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token    
    

    def data_collator(batch):        
        batch_dict = defaultdict(list)
        for example in batch:
            problem = example['problem']
            solution = example['solution']
            input_str = f"Question: {problem}\n\nSolution: {solution}"
            batch_dict['full_text'].append(input_str)
        
        model_inputs = tokenizer(
            batch_dict['full_text'], 
            padding=True, 
            truncation=True,            
            return_tensors='pt'
        )

        labels = model_inputs['input_ids'].clone()
                
        for idx, text in enumerate(batch_dict['full_text']):            
            solution_start = text.find('Solution:')
            solution_text_len = len('Solution:') + 1
            solution_tokens = tokenizer(text[:solution_start + solution_text_len], add_special_tokens=False)
            labels[idx, :len(solution_tokens['input_ids'])] = -100

        output_batch = {
            'input_ids': model_inputs['input_ids'],
            'attention_mask': model_inputs['attention_mask'],
            'labels': labels
        }        
        return output_batch

    common_kwargs = dict(
        collate_fn=data_collator,
        shuffle=True,        
        drop_last=True,
        pin_memory=True,
        num_workers=1,    # avoid weird shuffling issues
    )

    ds_train = ds['train']
    ds_test = ds['test']    

    train_dataloader = torch.utils.data.DataLoader(
        ds_train,
        batch_size=args.train_batch_size,
        **common_kwargs,
    )        
    test_dataloader = torch.utils.data.DataLoader(
        ds_test, 
        batch_size=args.test_batch_size,
        **common_kwargs,
    )
    num_epochs_steps = len(train_dataloader)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    num_training_steps = num_epochs_steps * args.epochs
    warmup_steps = int(num_training_steps * args.warmup_ratio)    
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps)    
    
    train_dataloader, test_dataloader = accelerator.prepare(
        train_dataloader, test_dataloader
    )
    
    def process_batch(batch):        
        return {
            'input_ids': batch['input_ids'].to(accelerator.device),
            'attention_mask': batch['attention_mask'].to(accelerator.device),
            'labels': batch['labels'].to(accelerator.device)
        }
    
    accelerator.init_trackers(
        project_name=args.wandb_project,
        config=OmegaConf.to_container(args, resolve=True), 
        init_kwargs=dict(wandb=dict(
            entity=args.wandb_entity,
            name=args.wandb_run_name,
        )),
    )
        
    total_steps = 0
    for epoch_num in range(args.epochs):
        model.train()
        for i, batch in tqdm.tqdm(enumerate(train_dataloader), total=num_epochs_steps, desc="training"):
            curr_progress_epoch = i / num_epochs_steps + epoch_num                                
            if args.debug:
                if total_steps % args.save_freq == 0 and total_steps > 0:
                    full_output_dir = os.path.join(args.output_dir, args.wandb_project, args.wandb_run_name, f"checkpoint_{total_steps}")
                    model_id = f"{args.wandb_project}_{args.wandb_run_name}_epoch{epoch_num}_checkpoint{total_steps}"
                    save_model(accelerator, model, tokenizer, full_output_dir, model_id)                    
            else:                
                if i % args.save_freq == 0 and i > 0:                    
                    full_output_dir = os.path.join(args.output_dir, args.wandb_project, args.wandb_run_name, f"checkpoint_{i}")
                    model_id = f"{args.wandb_project}_{args.wandb_run_name}_epoch{epoch_num}_checkpoint{i}"
                    save_model(accelerator, model, tokenizer, full_output_dir, model_id)                    

            total_steps += 1
                        
            if i % args.test_freq == 0:
                model.eval()
                with torch.no_grad():
                    test_losses = []
                    for _ in tqdm.tqdm(range(args.num_test_batches), total=args.num_test_batches, desc="Testing"):
                        test_batch = next(iter(test_dataloader))
                        test_batch = process_batch(test_batch)
                        
                        outputs = model(**test_batch)
                        test_loss = outputs.loss
                        test_losses.append(test_loss.item())

                    avg_test_loss = sum(test_losses) / len(test_losses)
                    accelerator.log({
                        'test/loss': avg_test_loss,
                        'epoch': curr_progress_epoch,
                    })
                model.train()
                        
            with accelerator.accumulate(model):
                train_batch = process_batch(batch)
                outputs = model(**train_batch)
                train_loss = outputs.loss                

                accelerator.backward(train_loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                lr = optimizer.param_groups[0]['lr']
                accelerator.print(f"Step {i} | Train Loss: {train_loss.float()} | LR: {lr}")
                accelerator.log({
                    'train/loss': train_loss.float(),
                    'train/lr': lr,
                    'epoch': curr_progress_epoch,
                })
                        
        if not args.debug:
            full_output_dir = os.path.join(args.output_dir, args.wandb_project, args.wandb_run_name, f"checkpoint-epoch{epoch_num}")        
            model_id = f"{args.wandb_project}_{args.wandb_run_name}_checkpoint-epoch{epoch_num}"
            save_model(accelerator, model, tokenizer, full_output_dir, model_id)
    

if __name__ == '__main__':
    main()