import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, hidden):
        super(SelfAttention, self).__init__()

        self.W = nn.Linear(hidden, 1)

    def forward(self, x):
        hidden = self.W(x)
        scores = hidden.bmm(hidden.transpose(1, 2))
        alpha = nn.functional.softmax(scores, dim=-1)
        attended = alpha.bmm(x)
        return attended