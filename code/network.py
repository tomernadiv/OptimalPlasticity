import torch
import torch.nn as nn
import itertools as it
import numpy as np

eps = 1e-18


class Network(nn.Module):
    def __init__(self, n, device=None):
        super(Network, self).__init__()
        self.n = n
        self.num_params = n*n - n
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        self.M = torch.tensor(list(it.product([0.0, 1.0], repeat=self.n))).to(self.device)  # shape (2^n, n)
        self.e = torch.zeros(2**self.n).to(self.device)
        self.e[-1] = 1
        self.pre, self.post = torch.where(torch.eye(self.n) == 0)    

    def params_to_mat(self, w):
        w = w.to(self.device)
        W = torch.zeros((self.n, self.n)).to(self.device)
        W[self.pre, self.post] = w
        return W    
    
    def forward(self, w, s):
        s = s.to(self.device)
        W = self.params_to_mat(w)
        if W.ndim == 1:
            W = self.params_to_mat(W)
        P_unnorm = torch.exp((self.M @ W @ self.M.T) + torch.ones_like(self.M) @ torch.diag(s) @ self.M.T)
        P = P_unnorm / (P_unnorm.sum(1)[:, None] + eps)
        P2 = P - torch.eye(2**self.n).to(self.device)
        P2[:, -1] = 1
        pi = torch.linalg.solve(P2.T, self.e)
        return pi


    
