import torch
from sklearn.manifold import MDS
import numpy as np
eps = 1e-8

def kl_divergence(P , Q, eps=1e-8):
    P = P + eps
    Q = Q + eps
    return (P * (P.log() - Q.log())).sum()

def js_divergence(P , Q, eps=1e-8):
    M = 0.5 * (P + Q)
    return 0.5 * (kl_divergence(P, M, eps) + kl_divergence(Q, M, eps))

def compute_functional_dist(net, w_traj, s):
    pi_traj = []
    for i in range(w_traj.shape[1]):
        w = torch.tensor(w_traj[:, i], dtype=torch.float32)
        pi = net(w, s).detach().cpu().numpy()
        pi_traj.append(torch.tensor(pi))
    return torch.stack(pi_traj)

def compute_djs_matrix(pi_traj):
    n_points = pi_traj.shape[0]
    djs_matrix = np.zeros((n_points, n_points))
    for i in range(n_points):
        for j in range(n_points):
            djs_matrix[i, j] = js_divergence(pi_traj[i], pi_traj[j], eps)
    return djs_matrix

def compute_mds_embedding(djs_matrix, n_components=2):
    mds = MDS(n_components=n_components, dissimilarity='precomputed', random_state=0)
    embedding = mds.fit_transform(djs_matrix)
    return embedding

def params_to_mat(param, n):
        mat = np.zeros((n, n))
        rows, cols = np.where(~np.eye(n, dtype=bool))
        mat[rows, cols] = param
        return mat

def mat_to_params(mat):
    n = mat.shape[0]
    params = np.zeros(n * (n - 1))
    rows, cols = np.where(~np.eye(n, dtype=bool))
    params[rows * (n - 1) + cols] = mat[rows, cols]
    return torch.tensor(params, dtype=torch.float32)