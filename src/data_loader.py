import torch
from torch.utils.data import Dataset, DataLoader
import glob
import json
from configs.config import Config

class QA_Dataset(Dataset):
    def __init__(self, files):
        self.files = files
        self.count = 0
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        self.count += 1
        print("Processing {}/{} files...".format(self.count, len(self.files)))
        with open(self.files[idx], 'r') as inf:
            data = json.load(inf)
            ids = data['id']
            word2idx = torch.tensor(data['word2idx'])
            c_emb = torch.tensor(data['context_embeddings'])
            q_emb = torch.tensor(data['question_embeddings'])
            a_start = torch.tensor(data['start_id'])
            a_end = torch.tensor(data['end_id'])
            e_start = torch.tensor(data['evi_start_id'])
            e_end = torch.tensor(data['evi_end_id'])


            return ids, word2idx, c_emb, q_emb, a_start, \
                    a_end, e_start, e_end




            # return ids , c_emb, q_emb, a_start, a_end, e_start, e_end




