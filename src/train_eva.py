import torch
import torch.nn.functional as F
import torch.optim as optim
from model import QA_1
from configs.config import Config
import re
import string
from collections import Counter

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = QA_1(config=Config)
optimizer = torch.optim.Adam(model.parameters(), betas=(0.8,0.999), eps=10e-7, weight_decay=3*10e-7)


def normalize_answer(s):
    '''
    Performs a series of cleaning steps on the ground truth and
    predicted answer.
    '''

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    '''
    Returns maximum value of metrics for predicition by model against
    multiple ground truths.

    :param func metric_fn: can be 'exact_match_score' or 'f1_score'
    :param str prediction: predicted answer span by the model
    :param list ground_truths: list of ground truths against which
                               metrics are calculated. Maximum values of
                               metrics are chosen.


    '''
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)

    return max(scores_for_ground_truths)


def f1_score(prediction, ground_truth):
    '''
    Returns f1 score of two strings.
    '''
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    '''
    Returns exact_match_score of two strings.
    '''
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def epoch_time(start_time, end_time):
    '''
    Helper function to record epoch time.
    '''
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def train(model, train_dataset):
    print("Starting training ........")
    train_loss = 0.
    batch_count = 0

    for batch in train_dataset:

        if batch_count % 500 == 0:
            print(f"Starting batch: {batch_count}")
        batch_count += 1

        id ,c_emb, q_emb, a_start, a_end, e_start, e_end = batch

        # place data on GPU
        c_emb, q_emb, a_start, a_end, e_start, e_end  = c_emb.to(device), q_emb.to(device), \
                                                        a_start.to(device), a_end.to(device), e_start.to(device), \
                                                               e_end.to(device)
        print(a_start)
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

    return train_loss /len(train_dataset)


def evaluate(predictions):
    '''
    Gets a dictionary of predictions with question_id as key
    and prediction as value. The validation dataset has multiple
    answers for a single question. Hence we compare our prediction
    with all the answers and choose the one that gives us
    the maximum metric (em or f1).
    This method first parses the JSON file, gets all the answers
    for a given id and then passes the list of answers and the
    predictions to calculate em, f1.


    :param dict predictions
    Returns
    : exact_match: 1 if the prediction and ground truth
      match exactly, 0 otherwise.
    : f1_score:
    '''
    with open('./data/squad_dev.json', 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    dataset = dataset['data']
    f1 = exact_match = total = 0
    for article in dataset:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                total += 1
                if qa['id'] not in predictions:
                    continue

                ground_truths = list(map(lambda x: x['text'], qa['answers']))

                prediction = predictions[qa['id']]

                exact_match += metric_max_over_ground_truths(
                    exact_match_score, prediction, ground_truths)

                f1 += metric_max_over_ground_truths(
                    f1_score, prediction, ground_truths)

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    return exact_match, f1


def valid(model, valid_dataset):
    print("Starting validation .........")

    valid_loss = 0.

    batch_count = 0

    f1, em = 0., 0.

    predictions = {}

    for batch in valid_dataset:

        if batch_count % 500 == 0:
            print(f"Starting batch {batch_count}")
        batch_count += 1

        context, question, char_ctx, char_ques, label, ctx_text, ans, ids = batch

        context, question, char_ctx, char_ques, label = context.to(device), question.to(device), \
                                                        char_ctx.to(device), char_ques.to(device), label.to(device)

        with torch.no_grad():

            preds = model(context, question, char_ctx, char_ques)

            p1, p2 = preds

            y1, y2 = label[:, 0], label[:, 1]

            loss = F.nll_loss(p1, y1) + F.nll_loss(p2, y2)

            valid_loss += loss.item()

            batch_size, c_len = p1.size()
            ls = nn.LogSoftmax(dim=1)
            mask = (torch.ones(c_len, c_len) * float('-inf')).to(device).tril(-1).unsqueeze(0).expand(batch_size, -1,
                                                                                                      -1)
            score = (ls(p1).unsqueeze(2) + ls(p2).unsqueeze(1)) + mask
            score, s_idx = score.max(dim=1)
            score, e_idx = score.max(dim=1)
            s_idx = torch.gather(s_idx, 1, e_idx.view(-1, 1)).squeeze()

            for i in range(batch_size):
                id = ids[i]
                pred = context[i][s_idx[i]:e_idx[i] + 1]
                pred = ' '.join([idx2word[idx.item()] for idx in pred])
                predictions[id] = pred

    em, f1 = evaluate(predictions)
    return valid_loss / len(valid_dataset), em, f1