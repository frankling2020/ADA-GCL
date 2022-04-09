import math
from multiprocessing.sharedctypes import Value
from winreg import QueryInfoKey
from cv2 import transform
from numpy import degrees
import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.nn import global_add_pool  
import torch.nn.functional as F
from utils import degree_measure, make_gin_conv

class GConv(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(GConv, self).__init__()
        
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for i in range(num_layers):
            if i == 0:
                self.layers.append(make_gin_conv(input_dim, hidden_dim))
            else:
                self.layers.append(make_gin_conv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        project_dim = hidden_dim * num_layers
        self.project = torch.nn.Sequential(
            nn.Linear(project_dim, project_dim),
            nn.ReLU(inplace=True),
            nn.Linear(project_dim, project_dim)
        )

    def forward(self, x, edge_index, batch):
        z = x  # (N, in_dims)
        zs = []
        for conv, bn in zip(self.layers, self.batch_norms):
            z = conv(z, edge_index)   # (N, hidden_dims)
            z = F.relu(z)
            z = bn(z) # batch normalization
            zs.append(z)   # [num_layers: [N, hidden_dims]]
        gs = [global_add_pool(z, batch) for z in zs]
        # z: [N, num_layers * hidden_dims]
        # g: [num_layers * hidden_dims]
        z, g = [torch.cat(x, dim=1) for x in [zs, gs]]
        return z, g

    def reg_loss(self): # regularization function
        loss = torch.sum(self.project.weight.data**2)
        for conv in self.layers:
            for layer in conv:
                if isinstance(layer, nn.Linear):
                    loss += torch.sum(layer.weight.data**2)
        return loss


class ADA(nn.Module):
    def __init__(self, in_dims, hidden_dims, num_layers, edge_dims=4, tau=0.2, beta=1) -> None:
        super(ADA, self).__init__()
        
        self.in_dims = in_dims
        self.hidden_dims = hidden_dims
        self.num_layers = num_layers
        self.tau = tau
        self.edge_dims = edge_dims
        self.beta = beta
        
        # learnable trandform
        self.proj_heads = num_layers*hidden_dims
        
        
        # learnable: edge importance
        self.Q = nn.Linear(self.proj_heads+1, edge_dims)
        self.K = nn.Linear(self.proj_heads+1, edge_dims)
        self.V = nn.Linear(edge_dims, edge_dims)
        
        # feature masking/dropping
        self.N = nn.Linear(self.proj_heads, in_dims)
        
        
        
        # encoder
        self.encoder = GConv(in_dims, hidden_dims, num_layers)
        self.augment_encoder = GConv(in_dims, hidden_dims, num_layers)
        
        
        x  --> transform --> encoder/augment_encoder ---> mlp --> g1, g2

        # mlp
        self.mlp = nn.Sequential(
            nn.Linear(self.proj_heads, self.proj_heads), 
            nn.ReLU(), 
            nn.Linear(self.proj_heads, hidden_dims))
        
        self.augment_mlp = nn.Sequential(
            nn.Linear(self.proj_heads, self.proj_heads), 
            nn.ReLU(), 
            nn.Linear(self.proj_heads, hidden_dims))
        
        self.reset_parameters()
        
        

    def reset_parameters(self):
        for params in self.parameters():
            if params.requires_grad == True:
                if len(params.shape) > 1:
                    nn.init.xavier_normal_(params)
                else:
                    params.fill_(0.0)
                    
    def edge_weight(self, z, edge_index, edge_attr=None):
        """_summary_

        Args:
            z: (N, num_layers * hidden_dims)
            edge_index: (2, num_edges)
        """
        
        # x, edge_index: (N, C), (2, E)
        
        src, dst = edge_index[0], edge_index[1]
        # structure features
        degree_weights = degree_measure(edge_index).unsqueeze(dim=1)
        
        # (E, num_layers * hidden_dims)
        src_embedding =  torch.cat([z[src], degree_weights[src]], dim=1)
        dst_embedding = torch.cat([z[dst], degree_weights[dst]], dim=1)
        
        # (E, num_layers * hidden_dims + 1)
        src_q = self.Q(src_embedding)
        dst_k = self.K(dst_embedding)
        src_v = self.V(edge_attr)  # (E, C) --> (E, edge_dims)
        
        # (E, edge_dims)  --> (E, 1)
        edge_weight = src_q * dst_k
        edge_weight = (F.softmax(edge_weight, dim=1) * src_v).mean(dim=1)
        
        
        # (N, num_layers * hidden_dims) ---> (N, in_dims) == x.shape  
        # (1, in_dims)
        node_feat_weights = (self.N(z) / math.sqrt(self.hidden_dims)).mean()
        
        return edge_weight, node_feat_weights
    
    
    