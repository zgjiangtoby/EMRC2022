import torch
from torch.utils.data import Dataset, DataLoader
import glob
import json
from configs.config import Config

class QA_Dataset(Dataset):
    def __init__(self, data_path):
        self.files = glob.glob(data_path)
    def __len__(self):
        return len(self.files[:50])
    def __getitem__(self, idx):
        print("Processing {}/{} line....".format(idx, len(self.files)))
        with open(self.files[idx], 'r') as inf:
            data = json.load(inf)
            # out_dict = {}
            # out_dict['id'] = self.data['id']
            # out_dict['c_emb'] = self.data['context_embeddings']
            # out_dict['q_emb'] = self.data['question_embeddings']
            # out_dict['a_start'] = self.data['start_positions']
            # out_dict['a_end'] = self.data['end_positions']
            # out_dict['e_start'] = self.data['evi_start_positions']
            # out_dict['e_end'] = self.data['evi_end_positions']
            ids = data['id']
            c_emb = torch.tensor(data['context_embeddings'])
            q_emb = torch.tensor(data['question_embeddings'])
            a_start = torch.tensor(data['start_id'])
            a_end = torch.tensor(data['end_id'])
            e_start = torch.tensor(data['evi_start_id'])
            e_end = torch.tensor(data['evi_end_id'])


            return ids, c_emb, q_emb, a_start, \
                    a_end, e_start, e_end




            # return ids , c_emb, q_emb, a_start, a_end, e_start, e_end




