from data_loader import *
import argparse
from sklearn.model_selection import train_test_split
from configs.config import Config
import glob
import torch
from model import QA_1, FGM
from utils import train, predict
from transformers import AutoTokenizer
import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="help")
    parser.add_argument("--data_path", type=str, help="this is your data directory")
    parser.add_argument("--tokenizer_path", type=str, help="this is your model directory")
    parser.add_argument("--predict", default=False, type=str, help="if True, generating predictions")
    parser.add_argument("--model_path", type=str, help="this is your model directory")
    parser.add_argument("--pred_path", default=None, type=str, help="prediction data path")
    args = parser.parse_args()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    if args.predict:
        print("Start prediction......")
        model = QA_1(config=Config)
        model.load_state_dict(torch.load(args.model_path))

        test_files = glob.glob(args.pred_path+"*.json")
        test_dataset = QA_Dataset(test_files)
        test_loader = DataLoader(test_dataset, shuffle=False, batch_size=Config.batch_size)
        predict(model.to(device), test_loader, device, tokenizer, "./model_outputs.json")

    else:
        print("Start training......")
        data_files = glob.glob(args.data_path+"*.json")

        dataset = QA_Dataset(data_files)

        train_, val_ = train_test_split(dataset, test_size=.2, shuffle=True)
        print("{} files in training.....".format(len(train_)))
        print("{} files in validation.....".format(len(val_)))
        train_loader = DataLoader(train_, batch_size=Config.batch_size)
        val_loader = DataLoader(val_, batch_size=Config.batch_size)


        data_iter = tqdm.tqdm(enumerate(train_loader),
                              desc="EP_%s:%d" % ("train", 1 + 1),
                              total=len(train_loader),
                              bar_format="{l_bar}{r_bar}")

        model = QA_1(config=Config).to(device)
        # =====================
        fgm = FGM(model)
        # =====================
        optimizer = torch.optim.Adam(model.parameters(), betas=(0.8, 0.999), eps=10e-7, weight_decay=3 * 10e-7)

        trained_model = train(train_loader, model, fgm, optimizer, device, val_loader, tokenizer)

        torch.save(trained_model.state_dict(), "./QA_1.model")




