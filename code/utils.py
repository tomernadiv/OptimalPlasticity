import torch

eps = 1e-10

def kl_divergence(P , Q, eps=1e-10):
    P = P + eps
    Q = Q + eps
    return (P * (P.log() - Q.log())).sum()

def js_divergence(P , Q, eps=1e-8):
    M = 0.5 * (P + Q)
    return 0.5 * (kl_divergence(P, M, eps) + kl_divergence(Q, M, eps))