import numpy as np
import pickle
import json
import torch
from torch.utils.data import Dataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dictionary = pickle.load(open('./src/data/dictionary.pkl', mode='rb'))

def sentence2ids(sentence, max_len=60):
    if len(sentence) > max_len:
        return np.array([dictionary.get(char, 0) for char in sentence[:max_len]])
    else:
        return np.array([dictionary.get(char, 0) for char in sentence + [0] * (max_len - len(sentence))])

class DealDataset(Dataset):
    def __init__(self):
        pass

    def __getitem__(self, index):
        pass

    def __len__(self):
        return self.len


def train(train_loaderm, model, loss_fn, optim):
    pass