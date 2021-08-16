import torch
import torch.nn as nn
from layers import SelfAttention, Attention

# torch.nn.CrossEntropyLoss()
# torch.nn.Softmax()
# torch.nn.functional.softmax()

class EntityTyping(nn.Module):
    def __init__(self, input_size, hidden_size, vocab_size, type_num):
        super(EntityTyping, self).__init__()
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
        self.bn = nn.BatchNorm1d(60)
        self.linear = nn.Linear(160*4, 256)
        self.out = nn.Linear(256, type_num)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=2e-5, eps=1e-8)

        self.attention = Attention(hidden_size*2)

    def forward(self, text, start, end, kb_text):
        # 原句
        text_vec = self.embedding(text)
        bitext_vec, (h_n, c_n) = self.bilstm(text_vec)
        bitext_vec = self.bn(bitext_vec)
        bitext_vec = self.dropout(bitext_vec)
        # batch_size, 40, 160

        # kb
        kb_text_vec = self.embedding(kb_text)
        kb_bitext_vec, (h_n, c_n) = self.bilstm(kb_text_vec)
        kb_bitext_vec = self.bn(kb_bitext_vec)
        kb_bitext_vec = self.dropout(kb_bitext_vec)
        # batch_size, 200, 160

        entity_start, entity_end = [], []
        for id_in_batch, (s, e) in enumerate(zip(start, end)):
            entity_start.append(bitext_vec[id_in_batch, s, :])
            entity_end.append(bitext_vec[id_in_batch, e, :])
        entity = torch.hstack((
            torch.vstack(entity_start),
            torch.vstack(entity_end),
        ))
        # batch_size, 320

        text_att = self.attention(bitext_vec)
        kb_text_att = self.attention(kb_bitext_vec)
        concat = torch.hstack((text_att, entity, kb_text_att))
        out = self.linear(concat)
        out = self.out(out)
        return out
