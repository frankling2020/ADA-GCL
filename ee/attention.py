from turtle import forward
from matplotlib import projections
import torch
import torch.nn as nn
import math
from utils import normalize

class LearnableAttn(nn.Module):
    def __init__(self, in_dims:int, hidden_dims:int, alpha:float=1e-2, proj=True):
        super(LearnableAttn, self).__init__()
        
        self.in_dims = in_dims
        self.hidden_dims = hidden_dims
        self.alpha = alpha
        self.proj = proj

        if self.proj:
            self.Q = nn.Parameter(torch.zeros(in_dims, hidden_dims))
            self.K = nn.Parameter(torch.zeros(in_dims, hidden_dims))
            

    def batch_attn(self, q, k, batch_size = 1024):
        num_lines = q.shape[0]
        epoches = num_lines // batch_size + 1
        attn = []
        for x in range(epoches):
            btach_attn = (q[x*batch_size:(x+1)*batch_size] * k[x*batch_size:(x+1)*batch_size]).sum(dim=1)
            attn.append(btach_attn)
        return torch.cat(attn)

    def forward(self, q_emb, k_emb, pos):
        q = q_emb
        k = k_emb

        if self.proj:
            q = torch.mm(q, self.Q)
            k = torch.mm(k, self.K)
        
        attn = self.batch_attn(q, k)
        
        # pos is within [0, 1]
        scaled_attn = attn / math.sqrt(self.hidden_dims) 
        
        # initial attention is all the same 0.5
        return (torch.sigmoid(scaled_attn) + self.alpha * (pos - 0.5)).clamp_(min=0, max=1)
        