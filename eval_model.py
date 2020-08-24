# -*- encoding: utf-8 -*-
'''
@File    :   eval.py
@Time    :   2020/08/18 14:56:35
@Author  :   yangning 
'''

import os
import torch
from torch.utils.data import DataLoader,RandomSampler,SequentialSampler,TensorDataset
from torch.optim import AdamW
from data_process import collate_fn,load_examples
from models.lr_scheduler import get_linear_schedule_with_warmup
from common import SpanEntityScore,bert_extract_item,ProgressBar

def evaluate(args, model, tokenizer,writer):
    
    metric = SpanEntityScore(args.id2label)
    eval_output_dir = args.output_dir
    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)
    eval_features = load_examples(args,tokenizer, data_type='dev')
    print ("***** Running eval *****")
    
    eval_loss = 0.0
    nb_eval_steps = 0
    pbar = ProgressBar(n_total=len(eval_features), desc="Evaluating")
    for step, f in enumerate(eval_features):
        input_lens = f.input_len
        input_ids = torch.tensor([f.input_ids[:input_lens]], dtype=torch.long).to(args.device)
        input_mask = torch.tensor([f.input_mask[:input_lens]], dtype=torch.long).to(args.device)
        segment_ids = torch.tensor([f.segment_ids[:input_lens]], dtype=torch.long).to(args.device)
        start_ids = torch.tensor([f.start_ids[:input_lens]], dtype=torch.long).to(args.device)
        end_ids = torch.tensor([f.end_ids[:input_lens]], dtype=torch.long).to(args.device)
        subjects = f.subjects
        model.eval()
        with torch.no_grad():
            inputs = {"input_ids": input_ids, "attention_mask": input_mask,
                      "start_positions": start_ids,"end_positions": end_ids}
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = (segment_ids if args.model_type in ["bert", "xlnet"] else None)
            outputs = model(**inputs)
            tmp_eval_loss, start_logits, end_logits = outputs[:3]
            R = bert_extract_item(start_logits, end_logits)
            T = subjects
            metric.update(true_subject=T, pred_subject=R)
            if args.n_gpu > 1:
                tmp_eval_loss = tmp_eval_loss.mean()
            eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1
        pbar(step)

    eval_loss = eval_loss / nb_eval_steps
    eval_info, entity_info = metric.result()
    results = {f'{key}': value for key, value in eval_info.items()}
    results['loss'] = eval_loss
    print ("***** Eval results *****")
    info = "-".join([f' {key}: {value:.4f} ' for key, value in results.items()])
    print (info)

    for key, value in results.items():
        writer.add_scalar(f"Eval_{key}", value, args.eval_count)

    for key, value in entity_info.items():
        writer.add_scalar(f"Eval_class_{key}_f1", value['f1'], args.eval_count)

    for key in sorted(entity_info.keys()):
        print ("******* %s results ********"%key)
        info = "-".join([f' {key}: {value:.4f} ' for key, value in entity_info[key].items()])
        print (info)

    args.eval_count += 1    
    return results






