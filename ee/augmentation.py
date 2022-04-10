import torch
from torch_geometric.utils import subgraph

def drop_edge(edge_index, edge_attn, p: float):
    edge_prob = edge_attn.where(edge_attn < p, torch.ones_like(edge_attn))
    sel_mask = torch.bernoulli(edge_prob).to(torch.bool)
    
    return edge_index[:, sel_mask]


def mask_feature(x, node_attn, p:float):
    return x * node_attn.where(node_attn < p, torch.ones_like(node_attn)).unsqueeze(-1)


def drop_feature(x, w, p: float = 0.7):
    w = w.where(w < p, torch.ones_like(w) * p)

    drop_mask = torch.bernoulli(1. - w).to(torch.bool)

    x = x.clone()
    x[:, drop_mask] = 0.

    return x


def drop_node(edge_index: torch.Tensor, node_attn:torch.Tensor, keep_prob: float = 0.5):
    num_nodes = int(edge_index.max().item()) + 1
    probs = torch.tensor([keep_prob for _ in range(num_nodes)])
    dist = torch.distributions.Bernoulli(probs)

    subset = dist.sample().to(torch.bool).to(edge_index.device)
    edge_index, _ = subgraph(subset, edge_index, node_attn)

    return edge_index

