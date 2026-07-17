import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import time
import math
import os

class MLP(nn.Module):
    def __init__(self, in_features=2, hidden=64, num_experts=3, depth=4):
        super().__init__()
        layers = []
        d = in_features
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.Tanh()]
            d = hidden
        layers.append(nn.Linear(d, num_experts))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


### baseline comparison ###
class UnsharedLayer(nn.Module):

    def __init__(self, hidden=64, depth=2, r=16):
        super().__init__()
        layers = [nn.Linear(1, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, r)]
        self.net = nn.Sequential(*layers)

        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, val_and_id):
        return self.net(val_and_id)

class Unshared_Expert(nn.Module):

    def __init__(self, hidden=32, r=16, dim=5):
        super().__init__()
        self.r = r
        self.dim = dim
        self.nets = nn.ModuleList([
            UnsharedLayer(hidden=hidden, r=r) 
            for _ in range(dim)
        ])
        
    def forward(self, coords):
        outs = []
        for j in range(self.dim):
            col = coords[:, j:j+1]
            fj = self.nets[j](col)     # (N,r)
            outs.append(fj)
        prod = outs[0]
        for j in range(1, self.dim):
            prod = prod * outs[j]            # (N,r)
        u = prod.sum(dim=-1, keepdim=True) / self.r # (N,1)
        # u = prod.sum(dim=-1, keepdim=True) 
        return u

### our shared model ###
class SharedLayer(nn.Module):

    def __init__(self, hidden=64, depth=2, r=16):
        super().__init__()
        layers = [nn.Linear(2, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, r)]
        self.net = nn.Sequential(*layers)

        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, val_and_id):
        return self.net(val_and_id)

class Expert(nn.Module):

    def __init__(self, hidden=32, r=16, dim=5):
        super().__init__()
        self.r = r
        self.dim = dim
        self.shared = SharedLayer(hidden, r=r)

    def _eval_dim(self, coord_col: torch.Tensor, dim_id: int) -> torch.Tensor:
        dim_col = torch.full_like(coord_col, float(dim_id))
        pair = torch.cat([coord_col, dim_col], dim=-1)   # (N,2)
        return self.shared(pair)      
        

    def forward(self, coords):
        outs = []
        for j in range(self.dim):
            col = coords[:, j:j+1]           # (N,1)
            fj = self._eval_dim(col, j)      # (N,r)
            outs.append(fj)
        prod = outs[0]
        for j in range(1, self.dim):
            prod = prod * outs[j]            # (N,r)
        u = prod.sum(dim=-1, keepdim=True) / self.r # (N,1)
        return u

### Used for domain decomposition ###
class DomainMoE(nn.Module):
    def __init__(self, in_features=5, num_experts=3, expert_hidden=32, expert_rank=16, router_hidden=64, router_depth=2):
        super().__init__()
        self.num_experts = num_experts
        self.in_features = in_features
        self.router = MLP(in_features=self.in_features, hidden=64, num_experts=self.num_experts, depth=4)
        # self.experts = nn.ModuleList([Unshared_Expert(hidden=expert_hidden, r=expert_rank, dim=in_features) for _ in range(num_experts)])
        self.experts = nn.ModuleList([Expert(hidden=expert_hidden, r=expert_rank, dim=in_features) for _ in range(num_experts)])
        
    def forward(self, x):
        gates = self.router(x)           # (N,E)
        gates = F.softmax(gates, dim=-1)
        expert_outputs = []
        for i in range(self.num_experts):
            output = self.experts[i](x)
            expert_outputs.append(output)
        expert_outputs = torch.stack(expert_outputs, dim=-1)  # (N,1,E)
        u_pred = gates.unsqueeze(1) * expert_outputs
        u_pred = torch.sum(u_pred, dim=-1)      # (N,1)
        return u_pred, gates