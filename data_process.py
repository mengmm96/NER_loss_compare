# -*- encoding: utf-8 -*-
'''
@File    :   data_process.py
@Time    :   2020/08/18 13:49:29
@Author  :   yangning 
'''
import torch
import logging
import os
import json
from torch.utils.data import TensorDataset
from models.transformers import BertTokenizer

def collate_fn(batch):

    all_input_ids, all_input_mask, all_segment_ids, all_start_ids,all_end_ids,all_lens = map(torch.stack, zip(*batch))
    max_len = max(all_lens).item()
    all_input_ids = all_input_ids[:, :max_len]
    all_input_mask = all_input_mask[:, :max_len]
    all_segment_ids = all_segment_ids[:, :max_len]
    all_start_ids = all_start_ids[:,:max_len]
    all_end_ids = all_end_ids[:, :max_len]
    return all_input_ids, all_input_mask, all_segment_ids, all_start_ids,all_end_ids,all_lens

def convert_examples_to_features(examples,label_list,max_seq_length,tokenizer,
                                 cls_token_at_end=False,cls_token="[CLS]",cls_token_segment_id=1,
                                 sep_token="[SEP]",pad_on_left=False,pad_token=0,pad_token_segment_id=0,
                                 sequence_a_segment_id=0,mask_padding_with_zero=True,):

    label2id = {label: i for i, label in enumerate(label_list)}
    features = []
    for (ex_index, example) in enumerate(examples):
        textlist = example.text_a
        subjects = example.subject
        tokens = tokenizer.tokenize(textlist)
        start_ids = [0] * len(tokens)
        end_ids = [0] * len(tokens)
        subjects_id = []
        for subject in subjects:
            label = subject[0]
            start = subject[1]
            end = subject[2]
            start_ids[start] = label2id[label]
            end_ids[end] = label2id[label]
            subjects_id.append((label2id[label], start, end))
        
        special_tokens_count = 2
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[: (max_seq_length - special_tokens_count)]
            start_ids = start_ids[: (max_seq_length - special_tokens_count)]
            end_ids = end_ids[: (max_seq_length - special_tokens_count)]

        tokens += [sep_token]
        start_ids += [0]
        end_ids += [0]
        segment_ids = [sequence_a_segment_id] * len(tokens)
        if cls_token_at_end:
            tokens += [cls_token]
            start_ids += [0]
            end_ids += [0]
            segment_ids += [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            start_ids = [0]+ start_ids
            end_ids = [0]+ end_ids
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
        input_len = len(input_ids)
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            start_ids = ([0] * padding_length) + start_ids
            end_ids = ([0] * padding_length) + end_ids
        else:
            input_ids += [pad_token] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length
            segment_ids += [pad_token_segment_id] * padding_length
            start_ids += ([0] * padding_length)
            end_ids += ([0] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(start_ids) == max_seq_length
        assert len(end_ids) == max_seq_length

        features.append(InputFeature(input_ids=input_ids,
                                  input_mask=input_mask,
                                  segment_ids=segment_ids,
                                  start_ids=start_ids,
                                  end_ids=end_ids,
                                  subjects=subjects_id,
                                  input_len=input_len))
    return features

def get_entity_bios(seq,id2label):

    chunks = []
    chunk = [-1, -1, -1]
    for indx, tag in enumerate(seq):
        if not isinstance(tag, str):
            tag = id2label[tag]
        if tag.startswith("S-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[2] = indx
            chunk[0] = tag.split('-')[1]
            chunks.append(chunk)
            chunk = (-1, -1, -1)
        if tag.startswith("B-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[0] = tag.split('-')[1]
        elif tag.startswith('I-') and chunk[1] != -1:
            _type = tag.split('-')[1]
            if _type == chunk[0]:
                chunk[2] = indx
            if indx == len(seq) - 1:
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
    return chunks

def get_entity_bio(seq,id2label):

    chunks = []
    chunk = [-1, -1, -1]
    for indx, tag in enumerate(seq):
        if not isinstance(tag, str):
            tag = id2label[tag]
        if tag.startswith("B-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[0] = tag.split('-')[1]
            chunk[2] = indx
            if indx == len(seq) - 1:
                chunks.append(chunk)
        elif tag.startswith('I-') and chunk[1] != -1:
            _type = tag.split('-')[1]
            if _type == chunk[0]:
                chunk[2] = indx

            if indx == len(seq) - 1:
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
    return chunks

def get_entities(seq,id2label,markup = 'bios'):

    assert markup in ['bio','bios']
    if markup =='bio':
        return get_entity_bio(seq,id2label)
    else:
        return get_entity_bios(seq,id2label)

def load_examples(args,tokenizer,data_type):

    if data_type == 'train':
        examples = DataProcess().get_train_examples(args.data_dir)
    elif data_type == 'dev':
        examples = DataProcess().get_dev_examples(args.data_dir)
    else:
        examples = DataProcess().get_test_examples(args.data_dir)

    features = convert_examples_to_features(
        examples=examples,
        tokenizer = tokenizer,
        label_list = DataProcess().get_labels(),
        max_seq_length=args.max_seq_length,
        cls_token_at_end=False,
        pad_on_left=False,
        cls_token=tokenizer.cls_token,
        cls_token_segment_id = 0,
        sep_token=tokenizer.sep_token,
        pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
        pad_token_segment_id= 0,
    )   

    if data_type =='dev':return features
    
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_start_ids = torch.tensor([f.start_ids for f in features], dtype=torch.long)
    all_end_ids = torch.tensor([f.end_ids for f in features], dtype=torch.long)
    all_input_lens = torch.tensor([f.input_len for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_start_ids,all_end_ids,all_input_lens)
    return dataset        

class InputFeature(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, input_len, segment_ids, start_ids,end_ids, subjects):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_ids = start_ids
        self.input_len = input_len
        self.end_ids = end_ids
        self.subjects = subjects

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

class CNerTokenizer(BertTokenizer):
    def __init__(self, vocab_file, do_lower_case=False):
        super().__init__(vocab_file=str(vocab_file), do_lower_case=do_lower_case)
        self.vocab_file = str(vocab_file)
        self.do_lower_case = do_lower_case

    def tokenize(self, text):
        _tokens = []
        for c in text:
            if self.do_lower_case:
                c = c.lower()
            if c in self.vocab:
                _tokens.append(c)
            else:
                _tokens.append('[UNK]')
        return _tokens

class InputExample(object):
    """A single training/test example for token classification."""
    def __init__(self, guid, text_a, subject):
        self.guid = guid
        self.text_a = text_a
        self.subject = subject
    def __repr__(self):
        return str(self.to_json_string())
    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output
    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

class DataProcess(object):

    def get_train_examples(self, data_dir):
        return self._create_examples(self._read_json(os.path.join(data_dir, "train.json")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(self._read_json(os.path.join(data_dir, "dev.json")), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(self._read_json(os.path.join(data_dir, "test.json")), "test")

    def get_labels(self):
        return ["O", "address", "book","company",'game','government','movie','name','organization','position','scene']

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line['words']
            labels = line['labels']
            subject = get_entities(labels,id2label=None,markup='bios')
            examples.append(InputExample(guid=guid, text_a=text_a, subject=subject))
        return examples

    def _read_json(self,input_file):
        lines = []
        with open(input_file,'r',encoding='utf8') as f:
            for line in f:
                line = json.loads(line.strip())
                text = line['text']
                label_entities = line.get('label',None)
                words = list(text)
                labels = ['O'] * len(words)
                if label_entities is not None:
                    for key,value in label_entities.items():
                        for sub_name,sub_index in value.items():
                            for start_index,end_index in sub_index:
                                assert  ''.join(words[start_index:end_index+1]) == sub_name
                                if start_index == end_index:
                                    labels[start_index] = 'S-'+key
                                else:
                                    labels[start_index] = 'B-'+key
                                    labels[start_index+1:end_index+1] = ['I-'+key]*(len(sub_name)-1)
                lines.append({"words": words, "labels": labels})
        return lines







