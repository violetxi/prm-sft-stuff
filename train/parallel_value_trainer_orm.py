import os
import tqdm
import torch
import numpy as np
import random
import hydra
import logging

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
import warnings
warnings.filterwarnings("ignore")
import transformers
import datasets
transformers.logging.set_verbosity_error()
from debug import set_trace


PAD_TOKEN = '<|PAD|>'
STEP_TOKEN = " \n\n"

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    return seed


def get_dataset(dataset_name, tokenizer=None):
    if dataset_name == 'math-mc':
        ds = datasets.load_dataset('Asap7772/hendrycks-math-mc-llama-parallel-filtered')
    elif dataset_name == 'math-mc-sft':
        ds = datasets.load_dataset('Asap7772/hendrycks-math-mc-llama-sftnoic-parallel-filtered')
    elif dataset_name == 'math-mc-sft-5k':
        ds = datasets.load_dataset('Asap7772/hendrycks-math-mc-llama-sft-intermediate-parallel-filtered')
    elif dataset_name == 'numina-v1-blocks-1620':
        ds = datasets.load_dataset('RLAIF/Value-v1-NUMINA-V1-Blocks-Merged-1620-problems-step-len-filtered')
    elif dataset_name == 'numina-v1-blocks-2964':
        ds = datasets.load_dataset('RLAIF/Value-v1-NUMINA-V1-Blocks-Merged-2964-problems-step-len-filtered')
    elif dataset_name == 'numina-v1-blocks-3194':
        ds = datasets.load_dataset('RLAIF/Value-v1-NUMINA-V1-Blocks-Merged-3194-problems-step-len-filtered')
    elif dataset_name == 'numina-v1-blocks-debug':
        ds = datasets.load_dataset('violetxi/Value-v1-NUMINA-V1-Blocks-Merged-step-len-filtered-debug')
    else:
        raise ValueError(f"Dataset {dataset_name} not found")
    return ds


def stable_loss_kl(p_1, p_2, mask=None, eps=1e-9):
    # p_1: 
    p_1 = torch.clamp(p_1, eps, 1 - eps)
    p_2 = torch.clamp(p_2, eps, 1 - eps)
    
    log_term_1 = torch.log((1 - p_1 + eps) / (1 - p_2 + eps) + eps)
    log_term_2 = torch.log((p_1 * (1 - p_2 + eps)) / (p_2 * (1 - p_1 + eps)) + eps)
    
    kl = log_term_1 + p_1 * log_term_2
    return kl


def stable_loss_bce(p_1, p_2, eps=1e-9):
    # Clamp probabilities to avoid log(0)
    p_1 = torch.clamp(p_1, eps, 1 - eps)
    p_2 = torch.clamp(p_2, eps, 1 - eps)

    # Compute binary cross-entropy for each pair of predictions
    bce_loss = -(p_1 * torch.log(p_2) + (1 - p_1) * torch.log(1 - p_2))
    
    return bce_loss


def stable_loss_huber(p_1, p_2):
    return torch.nn.SmoothL1Loss()(p_1, p_2)


@hydra.main(version_base=None, config_path="config", config_name="parallel_value_trainer")
def main(args: DictConfig):
    logging.info(f"Running with config: {args}")
    set_seed(args.seed)
    if args.gradient_accumulation_steps > 1:
        plugin = GradientAccumulationPlugin(
            num_steps=args.gradient_accumulation_steps,
            # adjust_scheduler=True,
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
    
    if args.evaluate_math500_orm:
        ds_math500 = load_dataset('Asap7772/hendrycks-math-mc-llama-sft-regen-refactored', split='test')
        
        def map_fn(examples):
            return_dict = {k:[] for k in examples.keys()}
            for i in range(len(examples['attempts'])):
                if len(examples['attempts'][i]) < 128:
                    continue
                if len(examples['attempts_answers']) > 128:
                    # sample 128 and shorten
                    idxs = np.random.choice(len(examples['attempts'][i]), 128, replace=False)
                    examples['attempts'][i] = [examples['attempts'][i][idx] for idx in idxs]
                    examples['attempts_answers'][i] = [examples['attempts_answers'][i][idx] for idx in idxs]
                    examples['correct'][i] = [examples['correct'][i][idx] for idx in idxs]
                for k in examples.keys():
                    return_dict[k].append(examples[k][i])
            return return_dict
        ds_math500 = ds_math500.map(map_fn, batched=True, num_proc=os.cpu_count())
    else:
        ds_math500 = None
            
    value_args = dict(
        model_name=args.model_name,
        cache_dir=args.cache_dir,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto",
        accelerator=accelerator,
        flash_attn=args.use_flash_attn,        
        # special_tokens={'additional_special_tokens': [], 'pad_token': PAD_TOKEN},
    )
    model = ValueFunction(**value_args)
    tokenizer = model.tokenizer
    

    def data_collator(batch):
        batch_dict = defaultdict(list)
        for example in batch:
            for k, v in example.items():
                batch_dict[k].append(v)
        
        END_OF_STEP_TOKEN_ID = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(STEP_TOKEN))[0]        

        output_batch = {}
        # append STEP_TOKEN manually with spaces to ensure proper tokenization        
        output_solutions = []
        for solution_steps in batch_dict['solution_steps']:
            solution_str = ""
            for solution_step in solution_steps:
                solution_str += solution_step.strip() + STEP_TOKEN
            output_solutions.append(solution_str)
        
        output_batch['solution'] = tokenizer(output_solutions, padding=True, truncation=True, return_tensors='pt')        
        
        is_eos = (output_batch['solution']['input_ids'] == END_OF_STEP_TOKEN_ID)        
        rtg = torch.zeros_like(output_batch['solution']['input_ids'], dtype=torch.float32)
        undiscounted_rtg = torch.zeros_like(output_batch['solution']['input_ids'], dtype=torch.float32)        

        for i, (curr_rtg, curr_undiscounted_rtg) in enumerate(zip(batch_dict['rtgs'], batch_dict['undiscounted_rtgs'])):            
            if is_eos[i].sum() != len(curr_rtg):
                set_trace(accelerator)
                raise ValueError(f"RTG length mismatch: {is_eos[i].sum()} != {len(curr_rtg)}")
            if is_eos[i].sum() != len(curr_undiscounted_rtg):
                set_trace(accelerator)
                raise ValueError(f"Undiscounted RTG length mismatch: {is_eos[i].sum()} != {len(curr_undiscounted_rtg)}")

            index_rtg = 0
            for j, eos in enumerate(is_eos[i]):
                if eos:
                    rtg[i, j] = curr_rtg[index_rtg]
                    undiscounted_rtg[i, j] = curr_undiscounted_rtg[index_rtg]
                    index_rtg += 1
                
        last_eos_mask = torch.zeros_like(is_eos)
        for i in range(last_eos_mask.shape[0]):
            for j in range(last_eos_mask.shape[1] - 1, -1, -1):
                if is_eos[i, j] == 1:
                    last_eos_mask[i, j] = 1
                    break
                
        output_batch['rtgs'] = rtg[last_eos_mask]
        output_batch['undiscounted_rtgs'] = undiscounted_rtg[last_eos_mask]
        output_batch['rtg_mask'] = is_eos
        output_batch['orm_mask'] = last_eos_mask                
        return output_batch

    common_kwargs = dict(
        collate_fn=data_collator,
        shuffle=True,        
        drop_last=True,
        pin_memory=True,
        num_workers=1,    # avoid weird shuffling issues
    )

    if args.balance_dataset:    # balancing positive and negative examples in the dataset
        def filter_fn_positive(examples):
            return examples['is_correct']
         
        def filter_fn_negative(examples):            
            return [not x for x in examples['is_correct']]
        
        ds_splits = ds['train'].train_test_split(test_size=args.test_split, seed=args.seed)
        ds_train_positive = ds_splits['train'].filter(filter_fn_positive, batched=True, num_proc=os.cpu_count())
        ds_train_negative = ds_splits['train'].filter(filter_fn_negative, batched=True, num_proc=os.cpu_count())
        ds_test_positive = ds_splits['test'].filter(filter_fn_positive, batched=True, num_proc=os.cpu_count())
        ds_test_negative = ds_splits['test'].filter(filter_fn_negative, batched=True, num_proc=os.cpu_count())

        train_pos_dataloader = torch.utils.data.DataLoader(
            ds_train_positive, 
            batch_size=args.train_batch_size // 2,
            **common_kwargs,
        )

        train_neg_dataloader = torch.utils.data.DataLoader(
            ds_train_negative, 
            batch_size=args.train_batch_size // 2,
            **common_kwargs,
        )
        
        test_pos_dataloader = torch.utils.data.DataLoader(
            ds_test_positive, 
            batch_size=args.test_batch_size // 2,
            **common_kwargs,
        )
        
        test_neg_dataloader = torch.utils.data.DataLoader(
            ds_test_negative, 
            batch_size=args.test_batch_size // 2,
            **common_kwargs,
        )

        num_epochs_steps = min(len(train_pos_dataloader), len(train_neg_dataloader))

        def itr_fn(dataloader1, dataloader2, tokenizer, num_epochs_steps):
            step = 0
            while step < num_epochs_steps:
                for x, y in zip(dataloader1, dataloader2):                    
                    assert isinstance(x, dict) and isinstance(y, dict), f"Data must be a dictionary: {x}, {y}"
                    assert x.keys() == y.keys(), f"Keys must match: {x.keys()} != {y.keys()}"
                    new_batch = dict()
                    first_key = list(x.keys())[0]
                    shuffle_indices = torch.randperm(len(x[first_key]) + len(y[first_key]))
                    for k in x.keys():
                        v1, v2 = x[k], y[k]
                        assert type(v1) == type(v2), f"Type {type(v1)} != {type(v2)} for key {k}"

                        if isinstance(v1, torch.Tensor):
                            try:
                                new_batch[k] = torch.cat([v1, v2], dim=0)
                                # shuffle the batch
                                new_batch[k] = new_batch[k][shuffle_indices]
                            except Exception as e:
                                # sequence of different length, re-pad
                                max_length = max(v1.shape[1], v2.shape[1])                                
                                v1 = torch.nn.functional.pad(v1, (0, max_length - v1.shape[1]), value=0)
                                v2 = torch.nn.functional.pad(v2, (0, max_length - v2.shape[1]), value=0)                                
                                new_batch[k] = torch.cat([v1, v2], dim=0)
                                # shuffle the batch
                                new_batch[k] = new_batch[k][shuffle_indices]

                        elif isinstance(v1, list) or isinstance(v1, tuple):
                            new_batch[k] = v1 + v2
                            # shuffle the batch
                            new_batch[k] = new_batch[k][shuffle_indices]

                        elif isinstance(v1, transformers.tokenization_utils_base.BatchEncoding):                            
                            max_length = max(v1["input_ids"].shape[1], v2["input_ids"].shape[1])
                            v1 = tokenizer.pad(v1, padding="max_length", max_length=max_length, return_tensors="pt")
                            v2 = tokenizer.pad(v2, padding="max_length", max_length=max_length, return_tensors="pt")
                            # combine two batches
                            combined_input_ids = torch.cat([v1["input_ids"], v2["input_ids"]], dim=0)
                            combined_attention_mask = torch.cat([v1["attention_mask"], v2["attention_mask"]], dim=0)
                            # shuffle the ids and attention masks
                            combined_input_ids = combined_input_ids[shuffle_indices]
                            combined_attention_mask = combined_attention_mask[shuffle_indices]
                            new_batch[k] = {"input_ids": combined_input_ids, "attention_mask": combined_attention_mask}
                            combined_batch = BatchEncoding({
                                "input_ids": combined_input_ids, "attention_mask": combined_attention_mask
                                })
                            new_batch[k] = combined_batch

                        else:                            
                            raise ValueError(f"Type {type(v1[0])} not recognized")
                        
                    step += 1
                    yield new_batch
        
        train_pos_dataloader, train_neg_dataloader, test_pos_dataloader, test_neg_dataloader = accelerator.prepare(
            train_pos_dataloader, train_neg_dataloader, test_pos_dataloader, test_neg_dataloader
        )
        train_dataloader = itr_fn(train_pos_dataloader, train_neg_dataloader, tokenizer, num_epochs_steps)
        test_dataloader = itr_fn(test_pos_dataloader, test_neg_dataloader, tokenizer, num_epochs_steps)

    else: 
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
        # text
        text = batch['solution']
        
        # intermediate reward
        reward = batch['rtgs'] if args.discounted_rewards else batch['undiscounted_rtgs']
        
        # intermediate reward mask (including final reward)
        reward_mask = batch['rtg_mask']
        
        # final reward mask
        orm_mask = batch['orm_mask']        

        return text.to(accelerator.device), reward.to(accelerator.device), reward_mask.to(accelerator.device), orm_mask.to(accelerator.device)
    
    accelerator.init_trackers(
        project_name=args.wandb_project,
        config=OmegaConf.to_container(args, resolve=True), 
        init_kwargs=dict(wandb=dict(
            entity=args.wandb_entity,
            name=args.wandb_run_name,
        )),
    )

    if args.loss == 'kl':
        stable_loss = stable_loss_kl
    elif args.loss == 'bce':
        stable_loss = stable_loss_bce
    elif args.loss == 'huber':
        stable_loss = stable_loss_huber    
    else:
        raise ValueError(f"Loss {args.loss} not found")
        
    total_steps = 0
    for epoch_num in range(args.epochs):
        for i, batch in tqdm.tqdm(enumerate(train_dataloader), total=num_epochs_steps, desc="parallel_value_trainer"):
            curr_progress_epoch = i / num_epochs_steps + epoch_num            
            if args.debug:
                if total_steps % args.save_freq == 0 and total_steps > 0:
                    full_output_dir = os.path.join(args.output_dir, args.wandb_project, args.wandb_run_name, f"checkpoint_{total_steps}")
                    os.makedirs(full_output_dir, exist_ok=True)
                    model.save(full_output_dir)                
            else:                
                if i % args.save_freq == 0 and i > 0:
                    full_output_dir = os.path.join(args.output_dir, args.wandb_project, args.wandb_run_name, f"checkpoint_{i}")
                    os.makedirs(full_output_dir, exist_ok=True)
                    model.save(full_output_dir)

            total_steps += 1
            
            if i % args.test_freq == 0:
                with torch.no_grad():
                    all_stacked = defaultdict(list)
                    for _ in tqdm.tqdm(range(args.num_test_batches), total=args.num_test_batches, desc="Testing"):
                        test_batch = next(iter(test_dataloader))
                        test_text, test_reward, test_prm_mask, test_orm_mask = process_batch(test_batch)
                        logits = model(**test_text).logits.squeeze(-1)
                        
                        rewards = torch.sigmoid(logits)
                        
                        p_1, p_2 = test_reward, rewards                                                
                        test_loss = stable_loss(p_1, p_2)                        

                        masked_orm_loss = test_loss.mean()
                        masked_orm_reward = rewards.mean()
                        masked_orm_target_reward = test_reward.mean()
                        
                        mse = (rewards - test_reward) ** 2
                        masked_orm_mse = mse.mean()
                        
                        rounded = torch.round(rewards)
                        rounded_target = torch.round(test_reward)
                        test_accuracy = (rounded == rounded_target).float()
                        
                        masked_orm_accuracy = test_accuracy.mean()                        

                        all_stacked['test/orm_loss'].append(masked_orm_loss)                        
                        all_stacked['test/orm_reward'].append(masked_orm_reward)                        
                        all_stacked['test/orm_target_reward'].append(masked_orm_target_reward)                        
                        all_stacked['test/orm_accuracy'].append(masked_orm_accuracy)                        
                        all_stacked['test/orm_mse'].append(masked_orm_mse)                        
                                                
                        # now look at positive and negative examples                        
                        pos_mask = test_reward > 0.5
                        neg_mask = test_reward <= 0.5
                        
                        pos_orm_accuracy = test_accuracy[pos_mask].mean()
                        neg_orm_accuracy = test_accuracy[neg_mask].mean()
                        
                        pos_orm_mse = mse[pos_mask].mean()
                        neg_orm_mse = mse[neg_mask].mean()
                                                
                        all_stacked['test/pos_orm_accuracy'].append(pos_orm_accuracy)
                        all_stacked['test/neg_orm_accuracy'].append(neg_orm_accuracy)
                        all_stacked['test/pos_orm_mse'].append(pos_orm_mse)
                        all_stacked['test/neg_orm_mse'].append(neg_orm_mse)                        
                        
                        if args.evaluate_math500_orm:
                            pass
                    
                    for k, v in all_stacked.items():
                        all_stacked[k] = torch.stack(v).float().mean()
                    accelerator.log(all_stacked)
            
            with accelerator.accumulate(model):
                train_batch = process_batch(batch)                
                train_text, train_reward, train_prm_mask, train_orm_mask = train_batch                
                logits = model(**train_text).logits.squeeze(-1)

                rewards = torch.sigmoid(logits)
                p_1, p_2 = train_reward, rewards                
                train_loss = stable_loss(p_1, p_2)

                accelerator.backward(train_loss.mean())
                optimizer.step()
                scheduler.step()                
                optimizer.zero_grad()                                

                with torch.no_grad():
                    masked_orm_reward = rewards.mean()
                    masked_orm_target_reward = train_reward.mean()
                    
                    rounded = torch.round(rewards)
                    rounded_target = torch.round(train_reward)
                    train_accuracy = (rounded == rounded_target).float()                                        
                    
                    mse = (rewards - train_reward) ** 2
                    masked_orm_mse = mse.mean()
                    
                    # now look at positive and negative examples                    
                    pos_mask = train_reward > 0.5
                    neg_mask = train_reward <= 0.5
                    
                    pos_orm_accuracy = train_accuracy[pos_mask].mean()
                    neg_orm_accuracy = train_accuracy[neg_mask].mean()
                    
                    pos_orm_mse = mse[pos_mask].mean()
                    neg_orm_mse = mse[neg_mask].mean()                    
                                        
                lr = optimizer.param_groups[0]['lr']
                accelerator.print(f"Step {i} | Train Loss: {train_loss.float().mean()} | Train ORM Accuracy: {train_accuracy.float().mean()}")
                accelerator.log({
                    'train/lr': lr,                    
                    'train/orm_loss': train_loss.mean().float(),
                    'train/orm_reward': masked_orm_reward.float(),
                    'train/orm_target_reward': masked_orm_target_reward,
                    'train/orm_accuracy': train_accuracy.mean(),
                    'train/orm_mse': masked_orm_mse,
                    'train/pos_orm_accuracy': pos_orm_accuracy,
                    'train/neg_orm_accuracy': neg_orm_accuracy,
                    'train/pos_orm_mse': pos_orm_mse,
                    'train/neg_orm_mse': neg_orm_mse,
                    'epoch': curr_progress_epoch,
                })
                
        # save checkpoint
        if not args.debug:
            full_output_dir = os.path.join(args.output_dir, args.wandb_project, args.wandb_run_name, f"checkpoint-epoch{epoch_num}")        
            os.makedirs(full_output_dir, exist_ok=True)
            accelerator.print(f"Saving checkpoint to {full_output_dir}")
            model.save(full_output_dir)
    

if __name__ == '__main__':
    main()