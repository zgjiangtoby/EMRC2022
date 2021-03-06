import tqdm
import json
import argparse
from transformers import AutoTokenizer, AutoModel

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

def preprocess_function(examples, model, if_evidence=True):
    data_iter = tqdm.tqdm(enumerate(examples),
                          desc="%s" % ("Preprocessing data"),
                          total=len(examples),
                          bar_format="{l_bar}{r_bar}")

    # device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    print("Found device: {}".format(device))
    for i, example in data_iter:
        input = {}
        input['id'] = example['id']
        q_input = tokenizer(
            example['question'],
            max_length=512,
            truncation=True,
            padding="max_length",
            return_tensors='pt'
        ).to(device)
        c_input = tokenizer(
            example['context'],
            max_length=512,
            truncation=True,
            padding="max_length",
            return_offsets_mapping=True,
            return_tensors='pt'
        ).to(device)
        embedder = model.to(device)

        q_embedding = embedder(**q_input)
        offset_mapping = c_input.pop("offset_mapping")
        c_embedding = embedder(**c_input)

        input['question_embeddings'] = q_embedding['last_hidden_state'].cpu().detach().numpy().tolist()
        input['context_embeddings'] = c_embedding['last_hidden_state'].cpu().detach().numpy().tolist()

        answers = example["answer"]

        word2idx = c_input['input_ids'].cpu().detach().numpy().tolist()[0]
        input['word2idx'] = word2idx

        answer_span = tokenizer.encode(answers["text"])[1:-1]
        start_position, end_position = contains(answer_span, word2idx)
        input['start_id'] = start_position
        input['end_id'] = end_position

        if if_evidence:
            evidence_span = tokenizer.encode(example['evidence'])[1:-1]
            evi_start_position, evi_end_position = contains(evidence_span, word2idx)
            input['evi_start_id'] = evi_start_position
            input['evi_end_id'] = evi_end_position

        with open(args.output + "/{}.json".format(input['id']), 'w') as out_file:
                json.dump(input, out_file)

if __name__ == "__main__":
        parser = argparse.ArgumentParser(description="help")
        parser.add_argument("--data_path", type=str, help="this is your data directory")
        parser.add_argument("--model_path", type=str, help="this is your model directory")
        parser.add_argument("--output", type=str, help="this is where your output going")
        args = parser.parse_args()

        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        embedder = AutoModel.from_pretrained(args.model_path)
        data = data_loader(args.data_path)
        preprocess_function(data, embedder)
