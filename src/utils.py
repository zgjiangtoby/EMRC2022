import json
import tqdm
import torch
import torch.nn.functional as F
from configs.config import Config
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import nltk
import re

writer = SummaryWriter()

def train(train_loader, model, optimizer, device, val_loader=None, tokenizer=None):

    for e in range(Config.epoch):
        train_loss = 0
        data_iter = tqdm.tqdm(enumerate(train_loader),
                              desc="EP_%s:%d" % ("train", e + 1),
                              total=len(train_loader),
                              bar_format="{l_bar}{r_bar}")

        for idx, batch in data_iter:
            id, word2idx, c_emb, q_emb, a_start, a_end, e_start, e_end = batch

            # place data on GPU
            c_emb, q_emb, a_start, a_end, e_start, e_end = c_emb.to(device), q_emb.to(device), \
                                                           a_start.to(device), a_end.to(device), e_start.to(device), \
                                                           e_end.to(device)

            # forward pass, get predictions
            preds = model(c_emb, q_emb)

            start_pred, end_pred, start_evi_pred, end_evi_pred = preds

            # calculate loss
            loss = F.cross_entropy(start_pred, a_start) + F.cross_entropy(end_pred, a_end) + \
                   F.cross_entropy(start_evi_pred, e_start) + F.cross_entropy(end_evi_pred, e_end)

            writer.add_scalar("b_loss/train", loss.item(), idx + 1)
            # backward pass
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.clip)
            # update the gradients
            optimizer.step()

            # zero the gradients so that they do not accumulate
            optimizer.zero_grad()

            train_loss += loss.item()

        writer.add_scalar("epoch_loss/train", train_loss/len(train_loader), e + 1)
        valid(model, val_loader, device, tokenizer, e)


        return model


def valid(model, val_loader, device, tokenizer, epoch):
    '''
    Performs validation.
    '''
    print("Start validation...........")
    valid_loss = 0
    # puts the model in eval mode. Turns off dropout
    model.eval()

    data_iter = tqdm.tqdm(enumerate(val_loader),
                          desc="EP_%s:" % ("valid"),
                          total=len(val_loader),
                          bar_format="{l_bar}{r_bar}")

    a_f1_scores = []
    e_f1_scores = []

    a_em_scores = []
    e_em_scores = []
    for idx, batch in data_iter:

        id, word2idx, c_emb, q_emb, a_start, a_end, e_start, e_end = batch
        # place data on GPU
        c_emb, q_emb, a_start, a_end, e_start, e_end = c_emb.to(device), q_emb.to(device), \
                                                       a_start.to(device), a_end.to(device), e_start.to(device), \
                                                       e_end.to(device)
        with torch.no_grad():

            preds = model(c_emb, q_emb)

            start_pred, end_pred, start_evi_pred, end_evi_pred = preds

            # calculate loss
            loss = F.cross_entropy(start_pred, a_start) + F.cross_entropy(end_pred, a_end) + \
                   F.cross_entropy(start_evi_pred, e_start) + F.cross_entropy(end_evi_pred, e_end)

            valid_loss += loss.item()
            writer.add_scalar("b_loss/eval", loss.item(), idx + 1)

            # get the start and end index positions from the model preds

            a_pred_start, a_pred_end = get_pred_position(start_pred, end_pred, device)
            e_pred_start, e_pred_end = get_pred_position(start_evi_pred, end_evi_pred, device)
            for i in range(len(a_pred_end)):
                # sample = {}
                # out = {}
                a_pred_s = a_pred_start[i]
                a_pred_e = a_pred_end[i]
                e_pred_s = e_pred_start[i]
                e_pred_e = e_pred_end[i]

                a_truth_s = a_start[i].cpu().detach().numpy().astype(int)
                a_truth_e = a_end[i].cpu().detach().numpy().astype(int)
                e_truth_s = e_start[i].cpu().detach().numpy().astype(int)
                e_truth_e = e_end[i].cpu().detach().numpy().astype(int)

                answer_id_span = find_span(word2idx=word2idx[i], start=a_pred_s, end=a_pred_e)
                evi_id_span = find_span(word2idx=word2idx[i], start=e_pred_s, end=e_pred_e)
                truth_answer_span = find_span(word2idx=word2idx[i], start=a_truth_s, end=a_truth_e)
                truth_evi_span = find_span(word2idx=word2idx[i], start=e_truth_s, end=e_truth_e)

                answer_f1 = cal_f1_score(truth_answer_span, answer_id_span, tokenizer)
                evidence_f1 = cal_f1_score(truth_evi_span, evi_id_span, tokenizer)

                answer_em = cal_em_score(truth_answer_span, answer_id_span, tokenizer)
                evidence_em = cal_em_score(truth_evi_span, evi_id_span, tokenizer)

                writer.add_scalar("iter_answer_f1/eval", answer_f1, i + 1)
                writer.add_scalar("iter_evidence_f1/eval", evidence_f1, i + 1)
                writer.add_scalar("iter_answer_em/eval", answer_em, i + 1)
                writer.add_scalar("iter_evidence_em/eval", evidence_em, i + 1)

                a_f1_scores.append(answer_f1)
                e_f1_scores.append(evidence_f1)

                a_em_scores.append(answer_em)
                e_em_scores.append(evidence_em)

    writer.add_scalar("avg_answer_f1/eval", avg_scores(a_f1_scores), epoch + 1)
    writer.add_scalar("avg_evidence_f1/eval", avg_scores(e_f1_scores), epoch + 1)
    writer.add_scalar("avg_answer_em/eval", avg_scores(a_em_scores), epoch + 1)
    writer.add_scalar("avg_evidence_em/eval", avg_scores(e_em_scores), epoch + 1)

    return avg_scores(a_f1_scores), avg_scores(e_f1_scores), \
           avg_scores(a_em_scores), avg_scores(e_em_scores)

def predict(model, test_loader, device, tokenizer, output_path):
    print("{} files in evaluation.....".format(len(test_loader)))
    valid_loss = 0
    # puts the model in eval mode. Turns off dropout
    model.eval()

    predictions = {}
    data_iter = tqdm.tqdm(enumerate(test_loader),
                          desc="EP_%s:" % ("valid"),
                          total=len(test_loader),
                          bar_format="{l_bar}{r_bar}")
    for idx, batch in data_iter:

        id, word2idx, c_emb, q_emb, a_start, a_end, e_start, e_end = batch
        # place data on GPU
        c_emb, q_emb, a_start, a_end, e_start, e_end = c_emb.to(device), q_emb.to(device), \
                                                       a_start.to(device), a_end.to(device), e_start.to(device), \
                                                       e_end.to(device)
        with torch.no_grad():

            preds = model(c_emb, q_emb)

            start_pred, end_pred, start_evi_pred, end_evi_pred = preds

            a_pred_start, a_pred_end = get_pred_position(start_pred, end_pred, device)
            e_pred_start, e_pred_end = get_pred_position(start_evi_pred, end_evi_pred, device)
            for i in range(len(a_pred_end)):
                sample = {}
                out = {}
                a_pred_s = a_pred_start[i]
                a_pred_e = a_pred_end[i]
                e_pred_s = e_pred_start[i]
                e_pred_e = e_pred_end[i]


                answer_id_span = find_span(word2idx=word2idx[i], start=a_pred_s, end=a_pred_e)
                evi_id_span = find_span(word2idx=word2idx[i], start=e_pred_s, end=e_pred_e)

                a_span = span_decoder(answer_id_span,tokenizer)
                e_span = span_decoder(evi_id_span,tokenizer)

                sample['answer'] = a_span
                sample['evidence'] = e_span


                out[id[i]] = sample
                predictions.update(out)
        with open(output_path, "w") as outfile:
            json.dump(predictions, outfile)

def span_decoder(in_span, tokenizer):
    return remove_special_token(tokenizer.decode(in_span))

def avg_scores(in_list):
    return 100 * sum(in_list) / len(in_list)


def cal_f1_score(truth, prediction, tokenizer):
    f1_score = 0
    real_span = remove_special_token(tokenizer.decode(truth))
    pred_span = remove_special_token(tokenizer.decode(prediction))

    real_segs = mixed_segmentation(real_span, rm_punc=True)
    prediction_segs = mixed_segmentation(pred_span, rm_punc=True)

    lcs, lcs_len = find_lcs(real_segs, prediction_segs)
    if lcs_len != 0:
        f1_score = f1(real_segs, prediction_segs, lcs_len)

    return f1_score


def f1(truth, prediction, lcs_len):
    lcs_len = int(lcs_len)
    precision = 1.0 * lcs_len / len(prediction)
    recall = 1.0 * lcs_len / len(truth)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def get_pred_position(start_pred, end_pred, device):
    batch_size, c_len = start_pred.size()
    ls = nn.LogSoftmax(dim=1)
    mask = (torch.ones(c_len, c_len) * float('-inf')).to(device).tril(-1).unsqueeze(0).expand(batch_size, -1, -1)

    score = (ls(start_pred).unsqueeze(2) + ls(end_pred).unsqueeze(1)) + mask
    score, s_idx = score.max(dim=1)
    score, e_idx = score.max(dim=1)

    s_idx = torch.gather(s_idx, 1, e_idx.view(-1, 1)).squeeze()
    return s_idx.cpu().detach().numpy().tolist(), e_idx.cpu().detach().numpy().tolist()

def remove_special_token(in_str):
    txt = in_str.lower().split()
    special_token = ['[cls]', '[sep]', '[unk]']
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
            lst.append(input_id.cpu().detach().numpy().tolist())
            if ix == end + 1:
                break
    return lst

def mixed_segmentation(in_str, rm_punc=False):
    in_str = str(in_str).lower().strip()
    segs_out = []
    temp_str = ""
    sp_char = ['-',':','_','*','^','/','\\','~','`','+','=',
               'ï¼Œ','ã€‚','ï¼š','ï¼Ÿ','ï¼','â€œ','â€','ï¼›','â€™','ã€Š','ã€‹','â€¦â€¦','Â·','ã€',
               'ã€Œ','ã€','ï¼ˆ','ï¼‰','ï¼','ï½ž','ã€Ž','ã€']
    for char in in_str:
        if rm_punc and char in sp_char:
            continue
        if re.search(r'[\u4e00-\u9fa5]', char) or char in sp_char:
            if temp_str != "":
                ss = nltk.word_tokenize(temp_str)
                segs_out.extend(ss)
                temp_str = ""
            segs_out.append(char)
        else:
            temp_str += char

    #handling last part
    if temp_str != "":
        ss = nltk.word_tokenize(temp_str)
        segs_out.extend(ss)

    return segs_out


def remove_punctuation(in_str):
    in_str = str(in_str).lower().strip()

    sp_char = ['-',':','_','*','^','/','\\','~','`','+','=',
               'ï¼Œ','ã€‚','ï¼š','ï¼Ÿ','ï¼','â€œ','â€','ï¼›','â€™','ã€Š','ã€‹','â€¦â€¦','Â·','ã€',
               'ã€Œ','ã€','ï¼ˆ','ï¼‰','ï¼','ï½ž','ã€Ž','ã€']
    out_segs = []
    for char in in_str:
        if char in sp_char:
            continue
        else:
            out_segs.append(char)
    return ''.join(out_segs)

def cal_em_score(truth, prediction, tokenizer):
    em = 0
    ans_ = remove_punctuation(tokenizer.decode(truth))
    prediction_ = remove_punctuation(tokenizer.decode(prediction))
    if ans_ == prediction_:
        em = 1
    return em


# find longest common string
def find_lcs(s1, s2):
    m = [[0 for i in range(len(s2)+1)] for j in range(len(s1)+1)]
    mmax = 0
    p = 0
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i] == s2[j]:
                m[i+1][j+1] = m[i][j]+1
                if m[i+1][j+1] > mmax:
                    mmax=m[i+1][j+1]
                    p=i+1
    return s1[p-mmax:p], mmax