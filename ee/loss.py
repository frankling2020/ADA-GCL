import imp
import torch
from abc import ABC, abstractmethod

import numpy as np
import torch.nn.functional as F

from sample import *


# def _similarity(h1: torch.Tensor, h2: torch.Tensor):
#     h1 = F.normalize(h1)
#     h2 = F.normalize(h2)
#     return h1 @ h2.t()


# class Loss(ABC):
#     @abstractmethod
#     def compute(self, anchor, sample, pos_mask, neg_mask, *args, **kwargs) -> torch.FloatTensor:
#         pass

#     def __call__(self, anchor, sample, pos_mask=None, neg_mask=None, *args, **kwargs) -> torch.FloatTensor:
#         loss = self.compute(anchor, sample, pos_mask,
#                             neg_mask, *args, **kwargs)
#         return loss


# class InfoNCE(Loss):
#     def __init__(self, tau):
#         super(InfoNCE, self).__init__()
#         self.tau = tau

#     def compute(self, anchor, sample, pos_mask, neg_mask, *args, **kwargs):
#         sim = _similarity(anchor, sample) / self.tau
#         exp_sim = torch.exp(sim) * (pos_mask + neg_mask)
#         log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True))
#         loss = log_prob * pos_mask
#         loss = loss.sum(dim=1) / pos_mask.sum(dim=1)
#         return -loss.mean()


# def add_extra_mask(pos_mask, neg_mask=None, extra_pos_mask=None, extra_neg_mask=None):
#     if extra_pos_mask is not None:
#         pos_mask = torch.bitwise_or(
#             pos_mask.bool(), extra_pos_mask.bool()).float()
#     if extra_neg_mask is not None:
#         neg_mask = torch.bitwise_and(
#             neg_mask.bool(), extra_neg_mask.bool()).float()
#     else:
#         neg_mask = 1. - pos_mask
#     return pos_mask, neg_mask


# class DualBranchContrast(torch.nn.Module):
#     def __init__(self, loss: Loss, intraview_negs: bool = False, **kwargs):
#         super(DualBranchContrast, self).__init__()
#         self.loss = loss
#         self.sampler = SameScaleSampler(intraview_negs=intraview_negs)
#         self.kwargs = kwargs

#     def forward(self, g1=None, g2=None, batch=None,
#                 extra_pos_mask=None, extra_neg_mask=None):
#         assert g1 is not None and g2 is not None
#         anchor1, sample1, pos_mask1, neg_mask1 = self.sampler(
#             anchor=g1, sample=g2)
#         anchor2, sample2, pos_mask2, neg_mask2 = self.sampler(
#             anchor=g2, sample=g1)

#         pos_mask1, neg_mask1 = add_extra_mask(
#             pos_mask1, neg_mask1, extra_pos_mask, extra_neg_mask)
#         pos_mask2, neg_mask2 = add_extra_mask(
#             pos_mask2, neg_mask2, extra_pos_mask, extra_neg_mask)
#         l1 = self.loss(anchor=anchor1, sample=sample1,
#                        pos_mask=pos_mask1, neg_mask=neg_mask1, **self.kwargs)
#         l2 = self.loss(anchor=anchor2, sample=sample2,
#                        pos_mask=pos_mask2, neg_mask=neg_mask2, **self.kwargs)

#         return (l1 + l2) * 0.5

def sim(z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())
    
def batched_semi_loss(z1: torch.Tensor, z2: torch.Tensor, batch_size: int, beta: float = 1.0, tau: float = 0.2):
    # Space complexity: O(BN) (semi_loss: O(N^2))
    device = z1.device
    num_nodes = z1.size(0)
    num_batches = (num_nodes - 1) // batch_size + 1
    f = lambda x: torch.exp(x / tau)
    indices = torch.arange(0, num_nodes, device=device)
    losses = []
    
    # this can be used to batched training

    for i in range(num_batches):
        mask = indices[i * batch_size:(i + 1) * batch_size]
        refl_sim = f(sim(z1[mask], z1))  # [B, N]
        between_sim = f(sim(z1[mask], z2))  # [B, N]

        losses.append(-torch.log(between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                                    / (refl_sim.sum(1) + beta * between_sim.sum(1)
                                    - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))

    return torch.cat(losses)

def loss_fn(z1: torch.Tensor, z2: torch.Tensor, mean: bool = True, batch_size: int = 128, beta: float = 1.0, tau: float = 0.2):
    l1 = batched_semi_loss(z1, z2, batch_size, beta)
    l2 = batched_semi_loss(z2, z1, batch_size, beta)
    
    ret = (l1 + l2) * 0.5
    ret = ret.mean() if mean else ret.sum()

    return ret     