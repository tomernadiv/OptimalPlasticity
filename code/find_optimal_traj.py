import numpy as np
import torch
from scipy.integrate import solve_bvp
from utils import eps  # your existing small constant


def L_struct(v):
    return torch.norm(v, p=2)

def L_func(w, v, net, s):
    J = torch.autograd.functional.jacobian(lambda w_: net.forward(w_, s), w)
    Jv = J @ v
    diag = torch.diag_embed(1/(net.forward(w, s) + 1e-2))
    return torch.sqrt((Jv @ diag @ Jv) + eps)

def L_total(w, v, net, s, alpha):
    return alpha * L_struct(v) + (1 - alpha) * L_func(w, v, net, s)

def compute_acceleration(w, v, net, s, alpha, damp=1e-6):
    # w, v are 1D torch tensors of shape (d,)
    w = w.clone().detach().requires_grad_(True)
    v = v.clone().detach().requires_grad_(True)

    L = L_total(w, v, net, s, alpha)

    dL_dw, dL_dv = torch.autograd.grad(L, (w, v), create_graph=True)

    Av = torch.autograd.functional.jacobian(lambda v_: torch.autograd.grad(L, v, create_graph=True)[0], v)
    Gw = torch.autograd.functional.jacobian(lambda w_: torch.autograd.grad(L, w, create_graph=True)[0], w)
    
    b = dL_dw - Gw @ v
    A_reg = Av + damp * torch.eye(len(v))
    a = torch.linalg.solve(A_reg, b)
    return a.detach()

def EL_first_order_bvp(t, y, net, s, alpha):
    """
    y: shape (2*d, n_points)
    returns dy/dt: shape (2*d, n_points)
    """
    d, n_points = y.shape[0] // 2, y.shape[1]
    dydt = np.zeros_like(y)

    for i in range(n_points):
        w = torch.from_numpy(y[:d, i]).float()
        v = torch.from_numpy(y[d:, i]).float()
        # if first point, use v guess from next point
        if i == 0:
            v_guess = torch.from_numpy((y[:d, i+1] - y[:d, i]) / (t[i+1]-t[i] + eps)).float()
            v = v_guess
        a = compute_acceleration(w, v, net, s, alpha)
        dydt[:d, i] = v.numpy()
        dydt[d:, i] = a.numpy()
    return dydt

def bc(y0, yT, w0, wT):
    w0, wT = w0.numpy() if isinstance(w0, torch.Tensor) else w0, wT.numpy() if isinstance(wT, torch.Tensor) else wT
    return np.concatenate([y0[:len(w0)] - w0, yT[:len(wT)] - wT])

def solve_el_bvp(w0, wT, net, s, alpha=1.0, n_points=20, T=1.0):
    d = len(w0)
    t_mesh = np.linspace(0, T, n_points)
    
    # Initial guess y = [w; v], shape (2*d, n_points)
    y_guess = np.zeros((2*d, n_points))
    for i in range(d):
        y_guess[i, :] = w0[i] + (wT[i]-w0[i]) * t_mesh  # linear w
        y_guess[d+i, :] = (wT[i]-w0[i]) / T             # constant v guess

    sol = solve_bvp(lambda t, y: EL_first_order_bvp(t, y, net, s, alpha),
                    lambda y0, yT: bc(y0, yT, w0, wT),
                    t_mesh, y_guess, max_nodes=1000)
    return sol

