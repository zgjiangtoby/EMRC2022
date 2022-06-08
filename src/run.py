from torch.utils.data import DataLoader
from data_loader import *
import argparse
from sklearn.model_selection import KFold
from configs.config import Config
import itertools
import torch
from model import QA_1
import tqdm
import torch.nn.functional as F

# device = "cuda:0" if torch.cuda.is_available() else "cpu"
device = 'cpu'
dataset = QA_Dataset("/media/ye/文档/CMRC2022/preprocessed_data/*.json")
k_fold = KFold(n_splits=5)

for fold_n, (train_idx, test_idx) in enumerate(k_fold.split(dataset)):
    print("Fold : {}\r".format(fold_n+1))
    print("loading train data to Dataloader...\r")
    train_data = []
    for tr_d in itertools.islice(dataset, train_idx[0], train_idx[-1]):
        train_data.append(tr_d)
    train_loader = DataLoader(train_data, batch_size=Config.batch_size)
    print("loading test data to Dataloader...\r")
    test_data = []
    for td in itertools.islice(dataset, test_idx[0], test_idx[-1]):
        test_data.append(td)
    test_loader = DataLoader(test_data, batch_size=Config.batch_size)

    model = QA_1(config=Config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), betas=(0.8, 0.999), eps=10e-7, weight_decay=3 * 10e-7)

    for e in range(Config.epoch):
        data_iter = tqdm.tqdm(enumerate(test_loader),
                              desc="%s: " % ("Training"),
                              total=len(train_idx),
                              bar_format="{l_bar}{r_bar}")

        train_loss = 0
        batch_count = 0
        for idx, batch in data_iter:
            id, c_emb, q_emb, a_start, a_end, e_start, e_end = batch
            print(c_emb.size())
            # place data on GPU
            c_emb, q_emb, a_start, a_end, e_start, e_end = c_emb.to(device), q_emb.to(device), \
                                                           a_start.to(device), a_end.to(device), e_start.to(device), \
                                                           e_end.to(device)


            # forward pass, get predictions
            preds = model(c_emb, q_emb)

            start_pred, end_pred = preds

            # calculate loss
            loss = F.cross_entropy(start_pred, a_start) + F.cross_entropy(end_pred, a_end)

            # backward pass
            loss.backward()

            # update the gradients
            optimizer.step()

            # zero the gradients so that they do not accumulate
            optimizer.zero_grad()

            train_loss += loss.item()

            print(loss.item())








