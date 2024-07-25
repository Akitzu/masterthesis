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
    upper = np.concatenate((path[2*130:, 10].T.flatten(), path[:2*30, 11].T.flatten()))
    
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
    lower = np.concatenate((path[2*132:, 17].T.flatten(), path[:2, 18].T.flatten()))

    startconfig = start_config(1e-6, sol.x, lambda_s, -v_s, jitedreversed, neps, K)
    path = evolve(startconfig, Nint, jitedreversed, K)
    out = path.T.flatten()
    idx = np.concatenate(([0], np.where(np.abs((out[:-2:2] - out[2::2])) > 0.5)[0], [-1]))
    for i in range(len(idx)-1):
        ax.plot(out[2*idx[i]+2:2*idx[i+1]:2], out[2*idx[i]+3:2*idx[i+1]+1:2], linewidth=1, c='g', zorder=20)
    
    return fig, ax, lower, upper

if __name__ == "__main__":
    saving_folder = Path("../../standardmap").absolute()

    k = 0.97
    nint = 19
    # nint = 30

    fig, ax, lower, upper = plot(k, nint, nev=20, nx=30, ny=10, neps=200)
    
    loop = np.concatenate((upper, lower)).reshape(-1, 2)

    import matplotlib.path as mpath
    # Assuming loop is a Nx2 array where each row is a point (x, y)
    x_min, y_min = np.min(loop, axis=0)
    x_max, y_max = np.max(loop, axis=0)

    # Create a meshgrid within the bounds
    x = np.linspace(x_min, x_max, 100)
    y = np.linspace(y_min, y_max, 100)
    X, Y = np.meshgrid(x, y)
    points = np.vstack((X.ravel(), Y.ravel())).T

    # Create a Path object from loop
    polygon = mpath.Path(loop)

    # Find points inside the domain
    inside = polygon.contains_points(points)

    # Filter points to keep only those inside the domain
    inside_points = points[inside]

    nev = 15
    dom_ev = np.empty((nev, inside_points.shape[0], 2))
    dom_ev[0,:,:] = inside_points

    for i in range(1,nev):
        for j in range(dom_ev.shape[1]):
            dom_ev[i,j,:] = standardmap(dom_ev[i-1,j,:], k)

    # plot the domain evolution
    for ii, dom in enumerate(dom_ev):
        if ii == 0 or ii == 11:
            ax.set_xlim(0.4, 0.6)
            ax.set_ylim(0.9, 1.1)
        if ii > 0:
            dom_plt.set_alpha(0.2)
        if ii == 2:
            ax.set_xlim(0, 1)
            ax.set_ylim(0.5, 1.5)
        
        dom_plt = ax.scatter(*dom.T, marker='.', color='tab:blue', edgecolor='black', s=1, zorder=30, linewidth=0.1)
        ax.set_title(f'Standard map, K={k:.3f}, nint={ii}')
        fig.savefig(saving_folder / f"outgoing_{ii}.png")