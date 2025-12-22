import numpy as np
import torch
import torch.nn as nn
import itertools as it
import matplotlib.pyplot as plt
from torch.func import vmap, jvp  # Requires PyTorch 2.0+

from utils import eps, js_divergence
from network import Network
from sklearn.manifold import MDS




class ActionMinimizer:
    def __init__(self, net, s, alpha=0.5, initial_path=None, n_points=100, T=1.0, min_lr=1e-6, verbose=True):
        self.net = net
        self.s = s
        self.alpha = alpha
        self.n_points = n_points
        self.min_lr = min_lr
        self.T = T
        self.dt = T / (n_points - 1)
        self.initial_path = initial_path
        self.verbose = verbose

        self.f_net = lambda w: self.net.forward(w, self.s)

    def compute_energies_vectorized(self, w_traj, v_traj):
        L_struct = torch.sum(v_traj**2, dim=1) 
        batched_jvp = vmap(lambda w, v: jvp(self.f_net, (w,), (v,)), in_dims=(0, 0))
        
        pi_traj, Jv_traj = batched_jvp(w_traj, v_traj)
        
        L_func = 0.5 * torch.sum((Jv_traj**2) / (pi_traj + eps), dim=1)
        
        return L_struct, L_func

    def total_action(self, w_traj):
        v_traj = (w_traj[1:] - w_traj[:-1]) / self.dt
        w_eval = w_traj[:-1]
        
        L_s, L_f = self.compute_energies_vectorized(w_eval, v_traj)
        
        L_total = self.alpha * L_s + (1 - self.alpha) * L_f
        action = torch.sum(L_total) * self.dt
        
        return action, L_s, L_f

    def train(self, w0, wT, steps=1000, lr=0.01, patience=100):
        w0 = w0.to(self.net.device).detach()
        wT = wT.to(self.net.device).detach()
        
        # Initialize Path (Linear)
        t_grid = torch.linspace(0, 1, self.n_points).to(self.net.device).unsqueeze(1)

        if self.initial_path is not None:
            initial_path = self.initial_path.to(self.net.device).detach()
        initial_path = w0.unsqueeze(0) + t_grid * (wT.unsqueeze(0) - w0.unsqueeze(0))
        # noise = 0.1 * torch.randn_like(initial_path)
        # noise[0] = 0.0
        # noise[-1] = 0.0
        # initial_path = initial_path + noise
        
        path_inner = initial_path[1:-1].clone().detach().requires_grad_(True)
        
        optimizer = torch.optim.Adam([path_inner], lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=patience, factor=0.5, verbose=True)
        
        history = {'loss': [],
                   'L_func': [],
                   'L_struct': []}
        if self.verbose:
            print(f"--- Starting Action Minimization (Alpha={self.alpha}) ---")
        
        for i in range(steps):
            optimizer.zero_grad()
            
            full_path = torch.cat([w0.unsqueeze(0), path_inner, wT.unsqueeze(0)], dim=0)
            
            # L_s_vec and L_f_vec are the instantaneous energies at each step
            loss, L_s_vec, L_f_vec = self.total_action(full_path)
            
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_([path_inner], 1.0)
            optimizer.step()
            scheduler.step(loss)
            
            action_s = torch.sum(L_s_vec) * self.dt
            action_f = torch.sum(L_f_vec) * self.dt
            history['loss'].append(loss.item())
            history['L_struct'].append(action_s.item())
            history['L_func'].append(action_f.item())
            
            if self.verbose and (i % 100 == 0 or i == steps - 1):
                print(f"Step {i:04d}: Loss={loss.item():.5f} | L_struct ={action_s.item():.4f} | L_func ={action_f.item():.4f} | LR={optimizer.param_groups[0]['lr']:.6f}")
        
            if optimizer.param_groups[0]['lr'] < self.min_lr:
                print("Learning rate below minimum threshold. Stopping training.")
                break

        final_path = torch.cat([w0.unsqueeze(0), path_inner, wT.unsqueeze(0)], dim=0).detach()

        return final_path, history
    
    def check_euler_lagrange(self, w_traj):
        """
        Validates the Euler-Lagrange equations: d/dt (dL/dv) - dL/dw = 0
        Returns the residuals (errors) at each time step.
        """
        w_traj = w_traj.detach().requires_grad_(True)
        
        # 1. Re-calculate velocities from the trajectory
        # Note: v[t] corresponds to the interval between w[t] and w[t+1]
        v_traj = (w_traj[1:] - w_traj[:-1]) / self.dt
        
        # 2. Compute Scalar Lagrangian at each step
        # We need to do this carefully to allow gradients wrt w_traj
        L_s, L_f = self.compute_energies_vectorized(w_traj[:-1], v_traj)
        
        # Your specific Lagrangian definition
        L_densities = self.alpha * L_s + (1 - self.alpha) * L_f
        
        # 3. Compute Gradients (Force and Momentum)
        # We assume L = sum(L_densities) for grad purposes
        L_sum = L_densities.sum()
        
        # dL/dv (Momentum p)
        # We want grad of L wrt v_traj
        grads_v = torch.autograd.grad(L_sum, v_traj, create_graph=True)[0]
        
        # dL/dw (Generalized Force)
        # We want grad of L wrt w_traj (specifically the points w_eval)
        # Note: compute_energies uses w_traj[:-1] as the evaluation points
        grads_w = torch.autograd.grad(L_sum, w_traj, create_graph=True)[0]
        # grads_w has shape of w_traj. We only care about the internal points for EL.
        
        # 4. Compute Discrete Euler-Lagrange Error
        # Equation: (p[t] - p[t-1]) / dt = dL/dw[t]
        
        # Momentum rate of change (dp/dt)
        # We shift grads_v to align p[t] and p[t-1]
        p_current = grads_v[1:]  # p associated with v[t]
        p_prev = grads_v[:-1]    # p associated with v[t-1]
        p_dot = (p_current - p_prev) / self.dt
        
        # Force (dL/dw)
        # corresponds to internal points w[1] to w[-2]
        force = grads_w[1:-1]
        
        # 5. Residual = dp/dt - dL/dw
        # This should be zero if the path is optimal
        el_residuals = p_dot - force
        
        # Calculate magnitude of error
        error_norm = torch.norm(el_residuals.reshape(el_residuals.shape[0], -1), p=2, dim=1)
        
        return error_norm, el_residuals

    

class Solution:
    def __init__(self, alpha, t, y, loss_history, struct_loss_history, func_loss_history):
        self.d = y.shape[0] // 2
        self.alpha = alpha
        self.t = t
        self.w = y[:self.d, :]
        self.v = y[self.d:, :]
        self.loss_history = loss_history
        self.struct_loss_history = struct_loss_history
        self.func_loss_history = func_loss_history
        self.L_s = struct_loss_history[-1]
        self.L_f = func_loss_history[-1]

    def __repr__(self):
        return fr"Solution for alpha={self.alpha}), $L_{{struct}}$={self.L_s:.4f}, $L_{{func}}$={self.L_f:.4f}"




def solve_optimal_trajectory_torch(w0, wT, net, s, initial_path=None, alpha=0.5, n_points=100, T=1.0, steps=2000, lr=0.01, min_lr=1e-6):
    minimizer = ActionMinimizer(net, s, alpha=alpha, n_points=n_points, T=T, min_lr=min_lr, initial_path=initial_path)
    final_path, history = minimizer.train(w0, wT, steps=steps, lr=lr)
    
    w_traj = final_path
    v_traj = (w_traj[1:] - w_traj[:-1]) / minimizer.dt
    v_traj = torch.cat([v_traj, v_traj[-1].unsqueeze(0)], dim=0)
    
    y_out = torch.cat([w_traj.T, v_traj.T], dim=0).cpu().numpy()
    x_out = np.linspace(0, T, n_points)

    loss_history = history['loss']
    struct_loss_history = history['L_struct']
    func_loss_history = history['L_func']

    norms, _ = minimizer.check_euler_lagrange(w_traj)
    mean_error = torch.mean(norms).item()
    print(f"Mean Euler-Lagrange Residual Error: {mean_error:.6f}")

    plt.plot(norms.cpu().detach().numpy())
    plt.title("Euler-Lagrange Residual Norms Over Time")
    plt.xlabel("Time Step")
    plt.ylabel("Residual Norm")
    plt.show()
        
    return Solution(alpha, x_out, y_out, loss_history, struct_loss_history, func_loss_history)



def create_mds_embedding(sol, net, s):
    w_traj = torch.from_numpy(sol.w.T).float().to(net.device)
    
    with torch.no_grad():
        s_tens = s.to(net.device) if isinstance(s, torch.Tensor) else torch.from_numpy(s).to(net.device)
        pi_traj = net.forward(w_traj, s_tens)
        
    pi_np = pi_traj.cpu()
    
    djs_matrix = np.zeros((pi_np.shape[0], pi_np.shape[0]))
    for i in range(pi_np.shape[0]):
        for j in range(i+1, pi_np.shape[0]):
            djs = js_divergence(pi_np[i], pi_np[j])
            djs_matrix[i, j] = djs
            djs_matrix[j, i] = djs

    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    embedding = mds.fit_transform(djs_matrix)
    
    return embedding