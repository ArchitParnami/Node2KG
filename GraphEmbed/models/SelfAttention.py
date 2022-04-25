import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, D_in):
        super(SelfAttention, self).__init__()
        K = D_in
        self.linear1 = nn.Linear(D_in, K)
        self.linear2 = nn.Linear(D_in, K)
        self.linear3 = nn.Linear(D_in, D_in)
    
    def forward(self, x, relations):
        keys = self.linear1(x)
        query = self.linear2(x)
        logits = torch.matmul(query,torch.t(keys))
        logits = logits + relations
        values = self.linear3(x)
        output = torch.matmul(logits, values)
        return output


