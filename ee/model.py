import math
import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool
import torch.nn.functional as F
from utils import *

class GConv(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(GConv, self).__init__()
        
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # Initialize the layers
        for i in range(num_layers):
            if i == 0:
                self.layers.append(make_gin_conv(input_dim, hidden_dim))
            else:
                self.layers.append(make_gin_conv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        
        project_dim = hidden_dim * num_layers
        
        # add projection head
        self.project = torch.nn.Sequential(
            nn.Linear(project_dim, project_dim),
            nn.ReLU(inplace=True),
            nn.Linear(project_dim, project_dim)
        )
        self.reset_parameters()

    def reset_parameters(self):
        for params in self.parameters():
            if params.requires_grad and params.ndim > 1:
                nn.init.xavier_uniform_(params)

    def _no_grads(self):
        for params in self.parameters():
            params.requires_grad_(False)
    
    def forward(self, x, edge_index, batch):
        z = x.to(torch.float32)
        
        zs = []
        for conv, bn in zip(self.layers, self.batch_norms):
            z = conv(z, edge_index)
            z = F.relu(z)
            z = bn(z)
            zs.append(z)
            
        gs = [global_add_pool(z, batch) for z in zs]
        
        # z is the local view, and g is the global view
        z, g = [torch.cat(x, dim=1) for x in [zs, gs]]
        
        return z, g