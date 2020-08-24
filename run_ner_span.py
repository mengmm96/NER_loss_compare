# -*- encoding: utf-8 -*-
'''
@File    :   run_ner_span.py
@Time    :   2020/08/18 11:05:33
@Author  :   yangning 
'''

import argparse
import os
import json
import torch
import glob
from common import seed_everything
from data_process import DataProcess
from data_process import convert_examples_to_features,load_examples
from models.transformers import BertConfig
from models.bert_for_ner import BertSpanForNer
from models.transformers import WEIGHTS_NAME
from eval_model import evaluate
from data_process import CNerTokenizer
from train_model import train
import time
from tensorboardX import SummaryWriter

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--task_name',default='NER',type=str)
    parser.add_argument('--data_dir',default='./datasets/cluener',type=str)
    parser.add_argument('--model_type',default='bert',type=str)
    parser.add_argument('--display',default='./display',type=str)
    parser.add_argument('--pretrain_model_path',default='./pretrained_model/bert-base-uncased/',type=str,required=False)
    parser.add_argument('--output_dir',default='./output/',type=str)
    parser.add_argument('--markup',default='bios',type=str,choices=['bios','bio'])
    parser.add_argument('--loss_type',default='ghmc',choices=['lsr','focal','ce','ghmc'])
    parser.add_argument('--max_seq_length',default=128,type=int)
    parser.add_argument("--do_lower_case", default=True)
    parser.add_argument('--do_train',default=True)
    parser.add_argument('--do_eval',default=True)
    parser.add_argument('--do_predict',default=False)
    parser.add_argument('--per_gpu_train_batch_size',default=128,type=int)
    parser.add_argument('--gradient_accumulation_steps',type=int,default=1)
    parser.add_argument('--learning_rate',default=5e-5,type=float)
    parser.add_argument("--weight_decay",default=0.0,type=float)
    parser.add_argument('--adam_epsilon',default=1e-8,type=float)
    parser.add_argument('--max_grad_norm',default=1.0,type=float)
    parser.add_argument('--num_train_epochs',default=8.0,type=float)
    parser.add_argument('--warmup_steps',default=0,type=int)
    parser.add_argument('--logging_steps',type=int,default=50)
    parser.add_argument('--save_steps',type=int,default=50)
    parser.add_argument('--no_cuda',default=False)
    parser.add_argument('--overwrite_output_dir',default=True)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--fp16',default=True)
    parser.add_argument('--fp16_opt_level',type=str,default="O1")
    parser.add_argument('--local_rank',type=int,default=-1)
    parser.add_argument("--eval_count", type=int, default=0)                         
    
    args = parser.parse_args()

    if not os.path.exists(args.display):
        os.mkdir(args.display)

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    args.output_dir = args.output_dir + f'{args.model_type}'
    
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda',args.local_rank)
        args.n_gpu = 1    
    args.device = device 

    seed_everything(args.seed)
    process = DataProcess()
    label_list = process.get_labels()

    args.id2label = {i : label for i, label in enumerate(label_list)}
    args.label2id = {label : i for i ,label in enumerate(label_list)}
    num_labels = len(label_list)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier() 

    config_class, model_class, tokenizer_class = BertConfig,BertSpanForNer,CNerTokenizer
    config = config_class.from_pretrained(args.pretrain_model_path,num_labels = num_labels,loss_type = args.loss_type,soft_label = True)
    tokenizer = tokenizer_class.from_pretrained(args.pretrain_model_path,do_lower_case = args.do_lower_case)
    model = model_class.from_pretrained(args.pretrain_model_path,config = config)

    model.to(args.device)

    writer = SummaryWriter(log_dir=args.display + '/' + time.strftime('%m_%d_%H.%M', time.localtime()) + '_' + str(args.loss_type))

    if args.do_train:
        train_dataset = load_examples(args,tokenizer,data_type = 'train')
        global_step, train_loss = train(args, train_dataset, model, tokenizer,writer)

        model_to_save = (model.module if hasattr(model, "module") else model)  
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_vocabulary(args.output_dir)

        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        checkpoints = [args.output_dir]

        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split('/')[-1] if checkpoint.find('checkpoint') != -1 else ""
            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            result = evaluate(args, model, tokenizer,writer)
            if global_step:
                result = {"{}_{}".format(global_step, k): v for k, v in result.items()}
            results.update(result)

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            for key in sorted(results.keys()):
                writer.write("{} = {}\n".format(key, str(results[key])))


if __name__ == "__main__":
    main()


