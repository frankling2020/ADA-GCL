import math
import torch
import torch.nn as nn
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
        z = x  #  B, C, H, W  -->  B, C1, H, W    ->>> B, C2, H, W
        zs = []
        for conv, bn in zip(self.layers, self.batch_norms):
            z = conv(z, edge_index)
            z = F.relu(z)
            z = bn(z) # batch normalization
            zs.append(z)
        gs = [global_add_pool(z, batch) for z in zs]
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
    """
        Goal:
        1. use trainable graph augumentation inspired by Transformer and GraphCL
        2. use structure similar to RNN to forget some mutual information trained by the enhanced loss function
        3. resolve the loss up-and-down problems
        4. Updating the difficulty in the negative pairs: sampled neagative pairs inspired by MoCo and CuCo 
    """
    def __init__(self, in_dims, hidden_dims, num_layers, edge_dims=1, tau=0.2):
        super(ADA, self).__init__()

        self.in_dims = in_dims
        self.hidden_dims = hidden_dims
        self.num_layers = num_layers
        self.tau = tau

        self.proj_heads = num_layers*hidden_dims
        
        self.encoder = GConv(in_dims, hidden_dims, num_layers)
        self.augment_encoder = GConv(in_dims, hidden_dims, num_layers)
        
        for params in self.augment_encoder.parameters():
            params.requires_grad = False
        
        self.Q = nn.Linear(self.proj_heads+1, edge_dims, bias=False)
        self.K = nn.Linear(self.proj_heads+1, edge_dims, bias=False)
        self.V = nn.Linear(edge_dims, edge_dims, bias=False)
        
        self.N = nn.Linear(self.proj_heads, in_dims, bias=False)
        
        self.mlp = nn.Sequential(nn.Linear(self.proj_heads, self.proj_heads), nn.ReLU(), nn.Linear(self.proj_heads, hidden_dims))
        self.aug_mlp = nn.Sequential(nn.Linear(self.proj_heads, self.proj_heads), nn.ReLU(), nn.Linear(self.proj_heads, hidden_dims))
        self.reset_parameters()
        
    def reset_parameters(self):
        for params in self.parameters():
            if params.requires_grad and len(params.shape) > 1:
                nn.init.xavier_normal_(params)

    def edge_weight(self, z, edge_index, edge_attr):
        src, dst = edge_index[0], edge_index[1]
        
        degree_weights = degree_measure(edge_index).unsqueeze(1)
        
        src_embedding = torch.cat([z[src], degree_weights[src]], dim=1)
        dst_embedding = torch.cat([z[dst], degree_weights[dst]], dim=1)
        
        src_q = self.Q(src_embedding)
        dst_k = self.K(dst_embedding)
        src_v = self.V(edge_attr)
        
        edge_weight = src_q * dst_k
        edge_weight = edge_weight / math.sqrt(self.hidden_dims)
        edge_weight = (F.softmax(edge_weight, dim=1) * src_v).mean(dim=1)


        node_weighted = self.N(z)
        node_feat_weight = (self.N(z)/math.sqrt(self.hidden_dims)).mean()

        return edge_weight, node_feat_weight

    def drop_edge_weighted(self, edge_index, edge_weights, threshold: float = 0.7):
        edge_weights = (edge_weights.max() - edge_weights)/(edge_weights.max() - edge_weights.mean())
       
        sel_mask = edge_weights < threshold
        
        return edge_index[:, sel_mask[:, 0]]

    def forward(self, x, edge_index, batch):
        z, g = self.encoder(x, edge_index, batch)
        return z, g

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())
    
    def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor, batch_size: int, beta: float = 1.0):
        # Space complexity: O(BN) (semi_loss: O(N^2))
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []
        
        # this can be used to batched training

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
            between_sim = f(self.sim(z1[mask], z2))  # [B, N]

            losses.append(-torch.log(between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                                     / (refl_sim.sum(1) + beta * between_sim.sum(1)
                                        - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))

        return torch.cat(losses)

    def loss(self, z1: torch.Tensor, z2: torch.Tensor, beta=1, mean: bool = True, batch_size: int = 64):
        l1 = self.batched_semi_loss(z1, z2, batch_size, beta)
        l2 = self.batched_semi_loss(z2, z1, batch_size, beta)
        
        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        return ret     

    def reg_loss(self):
        loss = 0
        for params in self.parameters():
            if params.requires_grad and len(params.shape) > 1:
                loss += torch.sum(params**2)
        return loss

    @property
    def total_parameters(self):
        total = 0
        for params in self.parameters():
            total += params.nelement()
        return total
    
    def encoder_update(self, inverse_momentum=1e-3):
        for param, aug_param in zip(self.encoder.parameters(), self.augment_encoder.parameters()):
            aug_param = param.clone().detach()