from model import *
from utils import *
from augmentation import *
from attention import LearnableAttn
import copy

class ADA(nn.Module):
    """
        Goal:
        1. use trainable graph augumentation inspired by Transformer and GraphCL
        2. use structure similar to RNN to forget some mutual information trained by the enhanced loss function
        3. resolve the loss up-and-down problems
        4. Updating the difficulty in the negative pairs: sampled neagative pairs inspired by MoCo and CuCo 
    """
    def __init__(self, in_dims, hidden_dims, num_layers, edge_dims=None):
        super(ADA, self).__init__()

        # initialize the parameters
        self.in_dims = in_dims
        self.hidden_dims = hidden_dims
        self.num_layers = num_layers
        
        # define the edge_dims
        if edge_dims is None:
            self.edge_dims = hidden_dims
        else:
            self.edge_dims = edge_dims

        # helper parameter
        self.proj_heads = num_layers*hidden_dims
        
        # encoder and augment encoder
        self.encoder = GConv(in_dims, hidden_dims, num_layers)

        # attention mechanism for learnable augmentation
        self.attn_node = LearnableAttn(self.proj_heads, self.hidden_dims, proj=False)
        self.attn_edge = LearnableAttn(2*self.proj_heads, 2*self.proj_heads, proj=False)
        
        self.mlp = nn.Sequential(nn.Linear(self.proj_heads, self.proj_heads), nn.ReLU(), nn.Linear(self.proj_heads, hidden_dims))

    def reset_parameters(self):
        for layer in self.mlp:
            initialize(layer)

    def attn_scores(self, z, g, batch, edge_index):
        src, dst = edge_index[0], edge_index[1]
        
        g_emb = g[batch]
        
        q_emb = torch.cat([g_emb[src], g_emb[dst]], dim=1)
        k_emb = torch.cat([z[src], z[dst]], dim=1)
        
        # edge_pos, node_pos = get_pagerank_weights(edge_index)
        edge_pos, node_pos = degree_measure(edge_index, z.shape[0])
        
        # edge attention
        edge_attn = self.attn_edge(q_emb, k_emb, edge_pos)
        
        # node attention
        node_attn = self.attn_node(g_emb, z, node_pos)

        return edge_attn, node_attn

    def forward(self, x, edge_index, batch):
        z = x.to(torch.float32)
        z, g = self.encoder(z, edge_index, batch)
        return z, g


    def reg_loss(self):
        loss = 0
        for params in self.parameters():
            if params.requires_grad and len(params.shape) > 1:
                loss += torch.sum(params**2)
        return loss / self.total_parameters

    @property
    def total_parameters(self):
        total = 0
        for params in self.parameters():
            if params.requires_grad:
                total += params.nelement()
        return total
    

    def learnable_transform(self, x, edge_index, edge_attn, node_attn, drop_rate):
        x_aug = x
        edge = edge_index
        
        x_aug = mask_feature(x, node_attn, drop_rate)
        edge = drop_edge(edge, edge_attn, p=drop_rate)

        return x_aug, edge

