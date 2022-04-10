from matplotlib import projections
import torch
import torch.nn as nn
import math
from utils import normalize
import torch.nn.functional as F

class LearnableAttn(nn.Module):
    def __init__(self, in_dims:int, hidden_dims:int, alpha:float = 1e-2, proj:bool =True):
        super(LearnableAttn, self).__init__()
        
        self.in_dims = in_dims
        self.hidden_dims = hidden_dims if proj else in_dims
        self.alpha = alpha
        self.proj = proj

        if self.proj:
            self.Q = nn.Linear(in_dims, hidden_dims)
            self.K = nn.Linear(in_dims, hidden_dims)
            nn.init.xavier_normal_(self.Q.weight.data)
            self.Q.bias.data.fill_(0.0)
            nn.init.xavier_normal_(self.K.weight.data)
            self.K.bias.data.fill_(0.0)

        self.ln1 = nn.LayerNorm(in_dims)
        self.ln2 = nn.LayerNorm(in_dims)

    def batch_attn(self, q, k, batch_size = 1024):
        num_lines = q.shape[0]
        epoches = num_lines // batch_size + 1
        attn = []
        for x in range(epoches):
            btach_attn = (q[x*batch_size:(x+1)*batch_size] * k[x*batch_size:(x+1)*batch_size]).sum(dim=1)
            attn.append(btach_attn)
        return torch.cat(attn)

    def forward(self, q_emb, k_emb, pos):
        q = self.ln1(q_emb)
        k = self.ln2(k_emb)

        if self.proj:
            q = self.Q(q)
            k = self.K(k)
        
        attn = self.batch_attn(q, k)
        scaled_attn = attn / math.sqrt(self.hidden_dims) 

        # pos is within [0, 1] and add position information
        pos_attn = torch.sigmoid(scaled_attn) + self.alpha * (pos - 0.5)

        # initial attention is all the same 0.5
        return pos_attn.clamp_(min=0, max=1)
        