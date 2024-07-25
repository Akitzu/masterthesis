import numpy as np
from pathlib import Path
from Kscan import *
import matplotlib.pyplot as plt
plt.style.use('lateky')

def plot(K, Nint, nev = 200, nx=30, ny=15, dpi=300, neps=800, msize = 3):
    xi = np.linspace(0., 1, nx)
    yi = np.linspace(0.5, 1.5, ny)
    # xi = np.linspace(0., 2*np.pi, 10)
    Xi = np.meshgrid(xi, yi)
    Xi = np.vstack((Xi[0].flatten(), Xi[1].flatten())).T
    # Xi = np.vstack((np.linspace(0., 1, 10), np.linspace(0., 1, 10))).T

    Ev = np.empty(((nev,)+Xi.shape))
    Ev[0] = Xi
    Ev.shape
    for i in range(1, Ev.shape[0]):
        evolved = np.array([jitedmap(Ev[i-1,j,:], K) for j in range(Ev.shape[1])])
        Ev[i,:,:] = evolved

    fig, ax = plt.subplots(dpi=dpi)
    ax.set_xlabel(r'$\theta$', fontsize=14)
    ax.set_ylabel(r'$p$', fontsize=14)
    ax.set_aspect('equal')
    ax.set_xlim(0, 1)
    ax.set_ylim(0.5, 1.5)
    fig.set_size_inches(6, 6)

    for i in range(len(Ev[0,:,0])):
         ax.scatter(Ev[:, i, 0], Ev[:, i, 1], s=0.1, alpha=1., zorder = 10, c='black')
    # ax.axis('off')

    from scipy.optimize import root
    sol = root(fixedpoint, [0.5, 0.5], (K))

    jacobian = jiteddmap(jnp.array([0.5, 1.]), K)
    ax.scatter(sol.x[0], sol.x[1], marker='X', color='tab:orange', edgecolor='black', zorder=30)
  
    lambda_s, v_s, lambda_u, v_u = eig(jacobian)
    
    startconfig = start_config(1e-6, sol.x, lambda_u, v_u, jitedmap, neps, K)
    path = evolve(startconfig, Nint, jitedmap, K)
    out = path.T.flatten()
    idx = np.concatenate(([0], np.where(np.abs((out[:-2:2] - out[2::2])) > 0.5)[0], [-1]))
    for i in range(len(idx)-1):
        ax.plot(out[2*idx[i]+2:2*idx[i+1]:2], out[2*idx[i]+3:2*idx[i+1]+1:2], linewidth=1, c='r', zorder=20)

    startconfig = start_config(1e-6, sol.x, lambda_u, -v_u, jitedmap, neps, K)
    path = evolve(startconfig, Nint, jitedmap, K)
    out = path.T.flatten()
    idx = np.concatenate(([0], np.where(np.abs((out[:-2:2] - out[2::2])) > 0.5)[0], [-1]))
    for i in range(len(idx)-1):
        ax.plot(out[2*idx[i]+2:2*idx[i+1]:2], out[2*idx[i]+3:2*idx[i+1]+1:2], linewidth=1, c='r', zorder=20)
    
    startconfig = start_config(1e-6, sol.x, lambda_s, v_s, jitedreversed, neps, K)
    path = evolve(startconfig, Nint, jitedreversed, K)
    out = path.T.flatten()
    idx = np.concatenate(([0], np.where(np.abs((out[:-2:2] - out[2::2])) > 0.5)[0], [-1]))
    for i in range(len(idx)-1):
        ax.plot(out[2*idx[i]+2:2*idx[i+1]:2], out[2*idx[i]+3:2*idx[i+1]+1:2], linewidth=1, c='g', zorder=20)

    startconfig = start_config(1e-6, sol.x, lambda_s, -v_s, jitedreversed, neps, K)
    path = evolve(startconfig, Nint, jitedreversed, K)
    out = path.T.flatten()
    idx = np.concatenate(([0], np.where(np.abs((out[:-2:2] - out[2::2])) > 0.5)[0], [-1]))
    for i in range(len(idx)-1):
        ax.plot(out[2*idx[i]+2:2*idx[i+1]:2], out[2*idx[i]+3:2*idx[i+1]+1:2], linewidth=1, c='g', zorder=20)
    
    return fig, ax

if __name__ == "__main__":
    saving_folder = Path("../../standardmap").absolute()

    k = 0.97
    nint = 19
    # nint = 30

    fig, ax = plot(k, nint, nev=20, nx=30, ny=10, neps=200)
    ax.set_xlim(0.4, 0.6)
    ax.set_ylim(0.9, 1.1)

    nev = 30
    pt_ev_1 = np.empty((nev, 2))
    pt_ev_2 = np.empty_like(pt_ev_1)
    pt_ev_1[0, :] = [0.5374, 1]
    pt_ev_2[0, :] = [0.5373, 1]
    for i in range(1,nev):
        pt_ev_1[i, :] = standardmap(pt_ev_1[i-1,:], k)
        pt_ev_2[i, :] = standardmap(pt_ev_2[i-1,:], k)

    for ii, (p1, p2) in enumerate(zip(pt_ev_1, pt_ev_2)):
        if ii > 0:
            p1_plt.set_alpha(0.2)
            p2_plt.set_alpha(0.2)
        if ii == 2:
            ax.set_xlim(0, 1)
            ax.set_ylim(0.5, 1.5)
        
        p1_plt = ax.scatter(*p1, marker='p', color='tab:blue', edgecolor='black', s=120, zorder=30, linewidth=1)
        p2_plt = ax.scatter(*p2, marker='D', color='orangered', edgecolor='black', s=40, zorder=30, linewidth=1)
        ax.set_title(f'Standard map, K={k:.3f}, nint={ii}')
        fig.savefig(saving_folder / f"runningpoint_{ii}.png")