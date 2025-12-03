import numpy as np
import torch
from scipy.integrate import solve_bvp
from utils import eps  

def L_struct_energy(v):
    return 0.5 * torch.sum(v**2)

def L_func_energy(w, v, net, s):
    pi = net.forward(w, s)
    _, Jv = torch.autograd.functional.jvp(lambda w_: net.forward(w_, s), (w,), (v,))
    diag_inv = 1.0 / (pi + 1e-6)
    return 0.5 * (Jv * diag_inv * Jv).sum()

def L_total_energy(w, v, net, s, alpha):
    E_struct = L_struct_energy(v)
    E_func = L_func_energy(w, v, net, s)
    return alpha * E_struct + (1 - alpha) * E_func

def compute_acceleration(w, v, net, s, alpha, damp=1e-5):
    """
    Solves for acceleration 'a' using Euler-Lagrange:
    d/dt (dL/dv) = dL/dw => H_vv * a = dL_dw - H_vw * v
    """
    w = w.detach().requires_grad_(True)
    v = v.detach().requires_grad_(True)

    L = L_total_energy(w, v, net, s, alpha)

    # 1. Gradients
    grad_L = torch.autograd.grad(L, (w, v), create_graph=True)
    dL_dw = grad_L[0]
    dL_dv = grad_L[1]

    # 2. Compute RHS term: dL_dw - (nabla_w p) @ v
    # (nabla_w p) @ v is directional derivative of p wrt w along v
    _, H_vw_v = torch.autograd.functional.jvp(
        lambda w_: torch.autograd.grad(L_total_energy(w_, v, net, s, alpha), v, create_graph=True)[0],
        (w,),
        (v,)
    )
    rhs = dL_dw - H_vw_v

    # 3. Compute LHS Matrix: H_vv (Metric Tensor G)
    # G = d(dL_dv)/dv. Since L is quadratic in v, G is pos-def.
    G = torch.autograd.functional.jacobian(
        lambda v_: torch.autograd.grad(L_total_energy(w, v_, net, s, alpha), v_, create_graph=True)[0],
        v
    )
    
    # Regularize G for numerical stability
    G_reg = G + damp * torch.eye(len(v)).to(G.device)
    
    # 4. Solve linear system G * a = rhs
    a = torch.linalg.solve(G_reg, rhs)
    
    return a.detach()

def EL_first_order_bvp(t, y, net, s, alpha):
    d = y.shape[0] // 2
    n_points = y.shape[1]
    dydt = np.zeros_like(y)

    for i in range(n_points):
        w_np = y[:d, i]
        v_np = y[d:, i]
        
        w = torch.from_numpy(w_np).float().to(net.device)
        v = torch.from_numpy(v_np).float().to(net.device)

        # Handle start point velocity guess if v is zero
        if i == 0 and np.allclose(v_np, 0):
             v_guess = (y[:d, i+1] - y[:d, i]) / (t[i+1]-t[i] + eps)
             v = torch.from_numpy(v_guess).float().to(net.device)

        try:
            acc = compute_acceleration(w, v, net, s, alpha)
            dydt[:d, i] = v.cpu().numpy()
            dydt[d:, i] = acc.cpu().numpy()
        except RuntimeError:
            # Fallback for extreme singularities
            dydt[:d, i] = v.cpu().numpy()
            dydt[d:, i] = 0

    return dydt

def bc(y0, yT, w0, wT):
    w0_np = w0.cpu().numpy() if isinstance(w0, torch.Tensor) else w0
    wT_np = wT.cpu().numpy() if isinstance(wT, torch.Tensor) else wT
    return np.concatenate([y0[:len(w0_np)] - w0_np, yT[:len(wT_np)] - wT_np])

def solve_el_bvp(w0, wT, net, s, alpha=0.5, n_points=100, T=1.0):
    d = len(w0)
    t_mesh = np.linspace(0, T, n_points)
    
    w0_np = w0.cpu().numpy()
    wT_np = wT.cpu().numpy()
    
    y_guess = np.zeros((2*d, n_points))
    avg_v = (wT_np - w0_np) / T
    
    for i in range(d):
        y_guess[i, :] = np.linspace(w0_np[i], wT_np[i], n_points)
        y_guess[d+i, :] = avg_v[i]

    # Wrapper to freeze net, s, alpha
    fun = lambda t, y: EL_first_order_bvp(t, y, net, s, alpha)
    bc_fun = lambda ya, yb: bc(ya, yb, w0_np, wT_np)

    sol = solve_bvp(fun, bc_fun, t_mesh, y_guess, max_nodes=1000, tol=1e-2)
    return sol


