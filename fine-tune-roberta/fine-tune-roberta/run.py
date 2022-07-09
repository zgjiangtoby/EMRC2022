import tqdm
import json
import argparse
from transformers import AutoTokenizer
from datasets import Dataset
import torch
from transformers import DefaultDataCollator
from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
import collections
import numpy as np
from torch import nn
from datasets import load_metric



def xrange(x):
    return iter(range(x))

def contains(small, big):
    for i in xrange(len(big)-len(small)+1):
        for j in xrange(len(small)):
            if big[i+j] != small[j]:
                break
        else:
            return i, i+len(small)
    return 0, 0

def find_max_evidence(lst):
    count = 0
    new_str = None
    for sent in lst:
        if len(sent) > count:
            count = len(sent)
            new_str = sent
    return new_str

def find_max_answer(lst):
    count = 0
    new_dic = {}
    for idx, sent in enumerate(lst):
        if len(sent['text']) > count:
            count = len(sent['text'])
            new_dic['text'] = sent['text']
            new_dic['answer_start'] = sent['answer_start']
    return new_dic

def data_loader(path, if_evidence=True):
    new_data = []
    with open(path, 'r',encoding='utf-8') as inf:
        dataset = json.load(inf)
        for line in dataset['data']:
            line = line['paragraphs'][0]
            qas = line['qas']
            for q in qas:
                new_dict = {}
                new_dict['id'] = q['id']
                new_dict['question'] = q['question']

                new_dict['answer'] = find_max_answer(q['answers'])
                new_dict['context'] = line['context']
                if if_evidence:
                    new_dict['evidence'] = find_max_evidence(q['evidences'])
                new_data.append(new_dict)
    return new_data

def drcd_loader(path):
    new_data = []
    with open(path, 'r',encoding='utf-8') as inf:
        dataset = json.load(inf)
        for line in dataset:

            new_dict = {}
            new_dict['id'] = line['id']
            new_dict['question'] = line['question']

            new_dict['answer'] = line['answer']
            new_dict['context'] = line['context']
            try:
                new_dict['evidence'] = line['evidence']
            except:
                continue
            new_data.append(new_dict)

    return new_data

def preprocess_function(examples, tokenizer):
    data_iter = tqdm.tqdm(enumerate(examples),
                          desc="%s" % ("Preprocessing data"),
                          total=len(examples),
                          bar_format="{l_bar}{r_bar}")

    ids = []
    input_ids = []
    token_type_ids = []
    attention_mask = []
    start_positions = []
    end_positions = []
    evi_start_positions = []
    evi_end_positions = []
    for i, example in data_iter:
        input = {}
        input['id'] = example['id']
        inputs = tokenizer(
            example['question'],
            example["context"],
            max_length=512,
            truncation="only_second",
            padding="max_length",
            return_offsets_mapping=True,
            return_tensors='pt'
        ).to(device)

        word2idx = inputs['input_ids'].cpu().detach().numpy().tolist()[0]

        offset_mapping = inputs.pop("offset_mapping")
        char_start = example['answer']['answer_start']

        s = 0
        for idx, off in enumerate(offset_mapping[0]):
            o = off.cpu().detach().numpy().tolist()[0]
            if char_start == o:
                s = idx

        e = s + len(example['answer']['text'])
        # span = find_span(word2idx, s, e)
        # print("a: ",example['answer']['text'])
        # print("b: ", span_decoder(span,tokenizer))

        inputs['start_positions'] = s
        inputs['end_positions'] = e
        ids.append(example['id'])
        input_ids.append(inputs['input_ids'].cpu().detach().numpy().tolist()[0])
        token_type_ids.append(inputs['token_type_ids'].cpu().detach().numpy().tolist()[0])
        attention_mask.append(inputs['attention_mask'].cpu().detach().numpy().tolist()[0])
        start_positions.append(inputs['start_positions'])
        end_positions.append(inputs['end_positions'])


        evidence_span = tokenizer.encode(example['evidence'])[1:-1]
        evi_start_position, evi_end_position = contains(evidence_span, word2idx)
        inputs['evi_start_position'] = evi_start_position
        inputs['evi_end_position'] = evi_end_position
        evi_start_positions.append(evi_start_position)
        evi_end_positions.append(evi_end_position)

    return ids, input_ids, token_type_ids, attention_mask, \
           start_positions, end_positions, evi_start_positions, evi_end_positions


def get_pred_position(start_pred, end_pred, device):
    batch_size, c_len = start_pred.size()
    ls = nn.LogSoftmax(dim=1)
    mask = (torch.ones(c_len, c_len) * float('-inf')).to(device).tril(-1).unsqueeze(0).expand(batch_size, -1, -1)

    score = (ls(start_pred).unsqueeze(2) + ls(end_pred).unsqueeze(1)) + mask
    score, s_idx = score.max(dim=1)
    score, e_idx = score.max(dim=1)

    s_idx = torch.gather(s_idx, 1, e_idx.view(-1, 1)).squeeze()
    return s_idx.cpu().detach().numpy().tolist(), e_idx.cpu().detach().numpy().tolist()

def span_decoder(in_span, tokenizer):
    return remove_special_token(tokenizer.decode(in_span))

def remove_special_token(in_str):
    txt = in_str.lower().split()
    special_token = ['[cls]', '[sep]', '[unk]', '[pad]']
    out_segs = []
    for t in txt:
        if t in special_token:
            continue
        else:
            out_segs.append(t)

    return ''.join(out_segs)


def find_span(word2idx, start, end):
    lst = []
    for ix, input_id in enumerate(word2idx):
        if ix >= start:
            lst.append(input_id)
            if ix == end - 1:
                break
    return lst

if __name__ == "__main__":
        parser = argparse.ArgumentParser(description="help")
        parser.add_argument("--data_path", type=str, help="this is your data directory")
        parser.add_argument("--ans_model", type=str, help="this is your ans_model directory")
        parser.add_argument("--evi_model", type=str, help="this is your evi_model directory")
        parser.add_argument("--pred_data", type=str, help="this is your data directory")
        args = parser.parse_args()
        device = torch.device("cuda:0")
        print("Found device: {}".format(device))
        model = AutoModelForQuestionAnswering.from_pretrained(args.ans_model).to(device)
        model.to(device)
        model2 = AutoModelForQuestionAnswering.from_pretrained(args.evi_model).to(device)
        model2.to(device)

        tokenizer = AutoTokenizer.from_pretrained(args.ans_model)
        #data = data_loader(args.data_path)
        data =drcd_loader(args.data_path)
        pred_data = data_loader(args.pred_data)

        ids, input_ids, token_type_ids, attention_mask, \
        start_positions, end_positions, evi_start_positions, evi_end_positions = preprocess_function(data, tokenizer)

        p_ids, p_input_ids, p_token_type_ids, p_attention_mask, \
        p_start_positions, p_end_positions, p_evi_start_positions, p_evi_end_positions = preprocess_function(pred_data, tokenizer)

        dataset = Dataset.from_dict({"ids":ids, "input_ids": input_ids, "token_type_ids":token_type_ids,
                                        "attention_mask": attention_mask, "start_positions": start_positions,
                                        "end_positions": end_positions})
        dataset2 = Dataset.from_dict({"ids":ids, "input_ids": input_ids, "token_type_ids":token_type_ids,
                                        "attention_mask": attention_mask, "start_positions": evi_start_positions,
                                        "end_positions": evi_end_positions})

        pred_dataset = Dataset.from_dict({"ids":p_ids, "input_ids": p_input_ids, "token_type_ids":p_token_type_ids,
                                        "attention_mask": p_attention_mask, "start_positions": p_start_positions,
                                        "end_positions": p_end_positions})

        pred_dataset2 = Dataset.from_dict({"ids": p_ids, "input_ids": p_input_ids, "token_type_ids": p_token_type_ids,
                                          "attention_mask": p_attention_mask, "start_positions": p_evi_start_positions,
                                          "end_positions": p_evi_end_positions})

        data_collator = DefaultDataCollator()

        training_args = TrainingArguments(
            output_dir="./results",
            evaluation_strategy="no",
            learning_rate=2e-5,
            per_device_train_batch_size=4,
            num_train_epochs=3,
            weight_decay=0.01,
            no_cuda=False,
        )
        trainer = Trainer(
            model=model.to(device),
            args=training_args,
            train_dataset=dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,

        )

        trainer2 = Trainer(model=model2.to(device),
            args=training_args,
            train_dataset=dataset2,
            tokenizer=tokenizer,
            data_collator=data_collator,

        )

        trainer.train()
        print("start trainer2 \n")
        trainer2.train()


        predictions, _, _ = trainer.predict(pred_dataset)
        evi_predictions,_,_ = trainer2.predict(pred_dataset2)
        start_pred, end_pred = predictions
        start_evi_pred, end_evi_pred = evi_predictions

        pred = {}
        a_pred_start, a_pred_end = get_pred_position(torch.tensor(start_pred).to(device), torch.tensor(end_pred).to(device), device)
        e_pred_start, e_pred_end = get_pred_position(torch.tensor(start_evi_pred).to(device), torch.tensor(end_evi_pred).to(device), device)

        for i in range(len(a_pred_end)):
            sample = {}
            out = {}
            a_pred_s = a_pred_start[i]
            a_pred_e = a_pred_end[i]
            e_pred_s = e_pred_start[i]
            e_pred_e = e_pred_end[i]

            #answer_id_span = find_span(word2idx=dataset['input_ids'][i], start=a_pred_s, end=a_pred_e)
            #evi_id_span = find_span(word2idx=dataset['input_ids'][i], start=e_pred_s, end=e_pred_e)
            answer_id_span = find_span(word2idx=pred_dataset['input_ids'][i], start=a_pred_s, end=a_pred_e)
            evi_id_span = find_span(word2idx=pred_dataset2['input_ids'][i], start=e_pred_s, end=e_pred_e)
            a_span = span_decoder(answer_id_span, tokenizer)
            e_span = span_decoder(evi_id_span, tokenizer)
            sample['answer'] = a_span
            sample['evidence'] = e_span
            out[pred_dataset['ids'][i]] = sample
            pred.update(out)
        with open("output.json", "w") as outfile:
            json.dump(pred, outfile, ensure_ascii=False)

        trainer.save_model('./answer.model')
        trainer2.save_model('./evidence.model')
