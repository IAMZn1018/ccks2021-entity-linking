import torch
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

class Attention(nn.Module):
    def __init__(self, hidden):
        super(Attention, self).__init__()
        self.linear = nn.Linear(hidden, 1, bias=False)

    def forward(self, x, mask=None):
        weights = self.linear(x)
        if mask is not None:
            weights = weights.mask_fill(mask.unsqueeze(2) == 0, float('-inf'))
        alpha = torch.softmax(weights, dim=1)
        outputs = x * alpha
        outputs = torch.sum(outputs, dim=1)
        return outputs
