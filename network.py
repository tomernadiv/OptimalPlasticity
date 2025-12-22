import torch
import torch.nn as nn
import itertools as it
import numpy as np
from utils import eps

class Network(nn.Module):
    def __init__(self, n, device=None):
        super(Network, self).__init__()
        self.n = n
        self.num_params = n*n - n
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        
        self.M = torch.tensor(list(it.product([0.0, 1.0], repeat=self.n))).float().to(self.device)
        
        self.e = torch.zeros(2**self.n).to(self.device)
        self.e[-1] = 1
        
        pre, post = torch.where(torch.eye(self.n) == 0)
        self.basis_matrices = torch.zeros(self.num_params, self.n, self.n, device=self.device)
        self.basis_matrices[torch.arange(self.num_params), pre, post] = 1.0

    def params_to_mat(self, w):
        w = w.to(self.device)
        return torch.tensordot(w, self.basis_matrices, dims=([-1], [0]))
    
    def forward(self, w, s):
        s = s.to(self.device)
        W = self.params_to_mat(w)
                
        if W.ndim == 3:
            B = W.shape[0]
            
            term1 = self.M.unsqueeze(0) @ W @ self.M.T.unsqueeze(0) 
            term2 = self.M @ torch.diag(s) @ self.M.T 
            P_unnorm = torch.exp(term1 + term2.unsqueeze(0))
            
            div = P_unnorm.sum(dim=2, keepdim=True) + eps
            P = P_unnorm / div
            
            I = torch.eye(2**self.n, device=self.device).unsqueeze(0)
            P2 = P - I
            P2[:, :, -1] = 1.0 
            
            A = P2.transpose(1, 2)
            b = self.e.unsqueeze(0).expand(B, -1)
            
            pi = torch.linalg.solve(A, b)
            return pi

        else:
            term1 = self.M @ W @ self.M.T
            term2 = self.M @ torch.diag(s) @ self.M.T
            P_unnorm = torch.exp(term1 + term2)
            P = P_unnorm / (P_unnorm.sum(1)[:, None] + eps)
            
            P2 = P - torch.eye(2**self.n).to(self.device)
            P2[:, -1] = 1

            pi = torch.linalg.solve(P2.T, self.e)
            return pi


    
