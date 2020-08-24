# -*- encoding: utf-8 -*-
'''
@File    :   train.py
@Time    :   2020/08/18 14:56:23
@Author  :   yangning 
'''
import os
import torch
from torch.utils.data import DataLoader,RandomSampler,SequentialSampler,TensorDataset
from torch.optim import AdamW
from data_process import collate_fn
from models.lr_scheduler import get_linear_schedule_with_warmup
from eval_model import evaluate
import time
from common import ProgressBar

def train(args,train_dataset,model,tokenizer,writer):

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,
                                  collate_fn=collate_fn)

    train_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": args.weight_decay,},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=train_total)

    if os.path.isfile(os.path.join(args.pretrain_model_path, "optimizer.pt")) and os.path.isfile(
        os.path.join(args.pretrain_model_path, "scheduler.pt")):
        optimizer.load_state_dict(torch.load(os.path.join(args.pretrain_model_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.pretrain_model_path, "scheduler.pt")))
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    print ("***** Running training *****")

    global_step = 0
    steps_trained_in_current_epoch = 0

    if os.path.exists(args.pretrain_model_path) and "checkpoint" in args.pretrain_model_path:
        global_step = int(args.pretrain_model_path.split("-")[-1].split("/")[0])
        epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

    train_loss, logging_loss = 0.0, 0.0
    model.zero_grad()

    for _ in range(int(args.num_train_epochs)):
        pbar = ProgressBar(n_total=len(train_dataloader), desc='Training')
        for step, batch in enumerate(train_dataloader):
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1],
                      "start_positions": batch[3],"end_positions": batch[4]}
            
            inputs["token_type_ids"] = (batch[2] if args.model_type in ["bert"] else None)
            outputs = model(**inputs)
            loss = outputs[0] 

            writer.add_scalar("Train_loss", loss.item(), step)

            if args.n_gpu > 1:
                loss = loss.mean() 
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            pbar(step, {'loss': loss.item()})
            train_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scheduler.step()  
                optimizer.step()
                model.zero_grad()
                global_step += 1
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:               
                    if args.local_rank == -1:
                        evaluate(args, model, tokenizer,writer)
                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):os.makedirs(output_dir)
                    model_to_save = (model.module if hasattr(model, "module") else model) 
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    tokenizer.save_vocabulary(output_dir)
                    print ("Saving model checkpoint to %s", output_dir)
                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    
        print(" ")
        if 'cuda' in str(args.device):
            torch.cuda.empty_cache()
    return global_step, train_loss / global_step





