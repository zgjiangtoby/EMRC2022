from torch.utils.data import DataLoader
from data_loader import *
import argparse
from sklearn.model_selection import KFold
from configs.config import Config
import itertools
import torch
from model import QA_1
from utils import train, valid
from transformers import AutoTokenizer


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # device = 'cpu'
    dataset = QA_Dataset("/media/ye/文档/CMRC2022/preprocessed_data/*.json")
    k_fold = KFold(n_splits=5)

    parser = argparse.ArgumentParser(description="help")
    parser.add_argument("--model_path", type=str, help="this is your model directory")
    parser.add_argument("--output", type=str, help="this is where your output going")
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    for fold_n, (train_idx, test_idx) in enumerate(k_fold.split(dataset)):
        print("Fold : {}\r".format(fold_n+1))
        print("loading training data to Dataloader...\r")
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

        train(train_loader, model, optimizer, device)
        valid(model, test_loader, device, tokenizer)










