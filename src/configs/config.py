from transformers import AutoTokenizer, AutoModel
import torch


class Config(object):
    data_path = '/media/ye/文档/CMRC2022/data/expmrc-main/cmrc2018_evid.json'
    model_path = "/home/ye/chinese-roberta-wwm-ext-large"
    preprocessed_data_path = "/media/ye/文档/CMRC2022/preprocessed_data"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    embedder = AutoModel.from_pretrained(model_path)