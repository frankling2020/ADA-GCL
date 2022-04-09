import torch
import torch.nn as nn

from torch_geometric.utils import degree, subgraph
from torch_geometric.nn import GINConv

from torch_scatter import scatter

def initialize(layer):
    if isinstance(layer, nn.Linear):
        nn.init.xavier_uniform_(layer.weight.data)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)


def normalize(x):
    batch_mins, _ = torch.min(x, dim=-1, keepdim=True)
    batch_maxs, _ = torch.max(x, dim=-1, keepdim=True)
    
    normalized_batch_embedding = torch.div(x - batch_mins, batch_maxs - batch_mins)
    
    return normalized_batch_embedding


def make_gin_conv(in_dims, out_dims):
    return GINConv(
            nn.Sequential(nn.Linear(in_dims, out_dims), nn.ReLU(inplace=True), nn.Linear(out_dims, out_dims)),
            eps=0,
            train_eps=False,
        )


def degree_measure(edge_index, num_nodes=None):
    deg = degree(edge_index[1], num_nodes).to(torch.float32)
    deg = torch.log(deg)
    weights = normalize(deg)
    return weights[edge_index[1]], weights 


def get_pagerank_weights(edge_index, aggr: str = 'sink', k: int = 10):
    def _compute_pagerank(edge_index, damp: float = 0.85, k: int = 10):
        num_nodes = edge_index.max().item() + 1
        deg_out = degree(edge_index[0])
        x = torch.ones((num_nodes,)).to(edge_index.device).to(torch.float32)

        for _ in range(k):
            edge_msg = x[edge_index[0]] / deg_out[edge_index[0]]
            agg_msg = scatter(edge_msg, edge_index[1], reduce='sum')

            x = (1 - damp) * x + damp * agg_msg

        return x

    pv = _compute_pagerank(edge_index, k=k)
    pv = torch.log(pv.to(torch.float32))
    pv = normalize(pv)
    s_row = pv[edge_index[0]]
    s_col = pv[edge_index[1]]
    if aggr == 'sink':
        s = s_col
    elif aggr == 'source':
        s = s_row
    elif aggr == 'mean':
        s = (s_col + s_row) * 0.5
    else:
        s = s_col

    return s, pv
