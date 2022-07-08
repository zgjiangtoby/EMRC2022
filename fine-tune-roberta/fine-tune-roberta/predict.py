from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from datasets import Dataset
import torch
import tqdm
import json
import argparse
from torch import nn


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
    with open(path, 'r') as inf:
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

def preprocess_function(example, if_evidence=True):
    inputs = tokenizer(
        example['question'],
        example["context"],
        max_length=384,
        truncation="only_second",
        padding="max_length",
        return_tensors='pt'
    ).to(device)

    answers = example["answer"]

    word2idx = inputs['input_ids'].cpu().detach().numpy().tolist()[0]

    if if_evidence:
        evidence_span = tokenizer.encode(example['evidence'])[1:-1]
        evi_start_position, evi_end_position = contains(evidence_span, word2idx)
        inputs['start_positions'] = torch.tensor(evi_start_position).unsqueeze(0)
        inputs['end_positions'] = torch.tensor(evi_end_position).unsqueeze(0)
    else:
        answer_span = tokenizer.encode(answers["text"])[1:-1]
        start_position, end_position = contains(answer_span, word2idx)
        inputs['start_positions'] = torch.tensor(start_position).unsqueeze(0)
        inputs['end_positions'] = torch.tensor(end_position).unsqueeze(0)

    return example['id'], inputs


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
            if ix == end:
                break
    return lst

if __name__ == "__main__":
        parser = argparse.ArgumentParser(description="help")
        parser.add_argument("--pred_data", type=str, help="this is your data directory")
        parser.add_argument("--ans_model", type=str, help="this is your model directory")
        parser.add_argument("--evi_model", type=str, help="this is your model directory")
        args = parser.parse_args()
        device = torch.device("cuda:0")

        ans_model = AutoModelForQuestionAnswering.from_pretrained(args.ans_model).to(device)
        ans_model.to(device)
        evi_model = AutoModelForQuestionAnswering.from_pretrained(args.evi_model).to(device)
        evi_model.to(device)

        tokenizer = AutoTokenizer.from_pretrained(args.evi_model)
        data = data_loader(args.pred_data)

        pred = {}
        for i in data:
            _, ans_data = preprocess_function(i, if_evidence=False)
            id, evi_data = preprocess_function(i, if_evidence=True)

            a_pred = ans_model(**ans_data)
            e_pred = evi_model(**evi_data)

            start_pred, end_pred = a_pred['start_logits'], a_pred['end_logits']
            start_evi_pred, end_evi_pred = e_pred['start_logits'], e_pred['end_logits']


            a_pred_start, a_pred_end = get_pred_position(torch.tensor(start_pred), torch.tensor(end_pred), device)
            e_pred_start, e_pred_end = get_pred_position(torch.tensor(start_evi_pred), torch.tensor(end_evi_pred),
                                                         device)


            sample = {}
            out = {}


            answer_id_span = find_span(word2idx=ans_data['input_ids'].cpu().detach().numpy().tolist()[0],
                                       start=a_pred_start, end=a_pred_end)
            evi_id_span = find_span(word2idx=evi_data['input_ids'].cpu().detach().numpy().tolist()[0],
                                    start=e_pred_start, end=e_pred_end)
            a_span = span_decoder(answer_id_span, tokenizer)
            e_span = span_decoder(evi_id_span, tokenizer)
            sample['answer'] = a_span
            sample['evidence'] = e_span
            out[id] = sample

            pred.update(out)

        with open("output_large.json", "w") as outfile:
            json.dump(pred, outfile, ensure_ascii=False)
