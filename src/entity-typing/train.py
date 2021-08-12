import numpy as np
import pickle
import json
import torch
import pandas as pd
from model import EntityTyping
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset, DataLoader

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dictionary = pickle.load(open('./src/data/dictionary.pkl', mode='rb'))

train_data = pd.read_csv('./src/data/entity_type/train.csv')
eval_data = pd.read_csv('./src/data/entity_type/dev.csv')
mention_type2id = {each: idx for idx, each in enumerate(
    train_data.mention_type.unique().tolist())}
pickle.dump(mention_type2id, open('./src/data/mention_type2id.pkl', mode='wb'))


def sentence2ids(sentence, max_len=60):
    if len(sentence) > max_len:
        return np.array([dictionary.get(char, 0) for char in sentence[:max_len]])
    else:
        return np.array([dictionary.get(char, 0) for char in sentence] + [0] * (max_len - len(sentence)))


class DealDataset(Dataset):
    def __init__(self, data):
        self.len = data.shape[0]
        self.text = data.text.map(sentence2ids).tolist()
        self.mention = data.mention.map(sentence2ids).tolist()
        self.start = data.offset.tolist()
        self.end = data.offset + data.mention.str.len().tolist()
        # self.end = self.end.tolist()
        self.label = data.mention_type.map(
            lambda x: mention_type2id[x]).tolist()

    def __getitem__(self, index):
        # return self.mention[index], self.start[index], self.end[index], self.text[index], self.label[index]
        return torch.LongTensor(self.mention[index]), torch.LongTensor(self.start[index]), torch.LongTensor(self.end[index]), torch.LongTensor(self.text[index]), torch.LongTensor(self.label[index])

    def __len__(self):
        return self.len


def evalute(eval_loader, model):
    preds, ys = [], []
    for step, (mention, start, end, text, y) in enumerate(eval_loader):
        mention, start, end, text, y = mention.to(device), start.to(
            device), end.to(device), text.to(device), y.to(device)

        pred = model(text, start, end)
        pred = torch.argmax(pred, dim=-1).view(-1).tolist()
        preds.extend(y.tolist())
    acc = accuracy_score(ys, preds)
    return acc


def train(train_loader, eval_loader, model):
    optimizer = model.optimizer
    loss_fn = model.criterion

    pre_acc = 0
    size = len(train_loader.dataset)
    for epoch in range(100):
        print('epoch: {}'.format(epoch))
        for step, (mention, start, end, text, y) in enumerate(train_loader):
            mention, start, end, text, y = mention.to(device), start.to(
                device), end.to(device), text.to(device), y.to(device)

            preds = model(text, start, end)
            loss = loss_fn(preds, y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step % 64 == 0:
                loss, current = loss.item(), step * len(mention)
                print(f"loss: {loss:>7f} [{current:>5d} / {size:>5d}]")

        train_acc = evalute(eval_loader, model)
        print(train_acc)

        if train_acc > pre_acc:
            pre_acc = train_acc
            torch.save(model, './h5/entity_typing.pkl')


train_data_set = DealDataset(train_data)
train_loader = DataLoader(train_data_set,
                          batch_size=1024,
                          shuffle=True
                          )

eval_data_set = DealDataset(eval_data)
eval_loader = DataLoader(eval_data_set,
                          batch_size=256,
                          shuffle=False
                          )

model = EntityTyping(
    input_size=80,
    hidden_size=80,
    vocab_size=len(dictionary),
    type_num=len(mention_type2id)
).to(device)

train(train_loader, eval_loader, model)