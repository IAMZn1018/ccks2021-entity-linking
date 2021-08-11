import torch
import torch.nn as nn
from layers import SelfAttention

# torch.nn.CrossEntropyLoss()
# torch.nn.Softmax()
# torch.nn.functional.softmax()

class EntityTyping(nn.Module):
    def __init__(self, input_size, hidden_size, vocab_size, type_num):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=hidden_size
        )
        self.bilstm = nn.LSTM(input_size,
                              hidden_size, 
                              batch_first=True, 
                              bidirectional=True)
        self.dropout = nn.Dropout(p=0.5)
        self.bn = nn.BatchNorm1d(self.hidden_size)
        self.linear = nn.Linear(2*self.hidden_size, 128)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=2e-5, eps=1e-8)

        self.attention = SelfAttention(hidden_size)

    def forward(self, sentence, entity_start):
        sentence_vec = self.embedding(sentence)
        sentence_vec = self.bn(sentence_vec)

        entity_start = []
        for id_in_batch, start in enumerate(entity_start):
            entity_start.append(sentence_vec[id_in_batch, start, :])
        entity = torch.vstack(entity_start)

        sentence_att = self.attention(sentence_vec)
        concat = torch.hstack((sentence_att, entity))
        out = self.linear(concat)
        return out