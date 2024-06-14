import numpy as np
import matplotlib.pyplot as plt
plt.style.use('lateky')
import jax.numpy as jnp
from jax import jacfwd

def standardmap(xy, K):
    ynew = xy[1] - K*jnp.sin(2*jnp.pi*xy[0])/(2*jnp.pi)
    xnew = jnp.mod(xy[0] + ynew, 1)
    return jnp.array([xnew, ynew])

def reversedmap(xy, K):
    xold = jnp.mod(xy[0] - xy[1], 1)
    yold = xy[1] + K*jnp.sin(2*jnp.pi*xold)/(2*jnp.pi)
    return jnp.array([xold, yold])

from jax import jit
jitedmap = jit(standardmap, static_argnums=1)
jitedreversed = jit(reversedmap, static_argnums=1)

dstandardmap = jacfwd(standardmap)
jiteddmap = jit(dstandardmap, static_argnums=1)


def eig(jacobian):
    """Compute the eigenvalues and eigenvectors of the jacobian and returns them in the order : stable, unstable."""
    eigRes = np.linalg.eig(jacobian)
    eigenvalues = np.abs(eigRes[0])

    # Eigenvectors are stored as columns of the matrix eigRes[1], transposing it to access them as np.array[i]
    eigenvectors = eigRes[1].T
    s_index, u_index = 0, 1
    if eigenvalues[0].real > eigenvalues[1].real:
        s_index, u_index = 1, 0

    return (
        eigenvalues[s_index],
        eigenvectors[s_index],
        eigenvalues[u_index],
        eigenvectors[u_index],
    )

def start_config(epsilon, rfp, eigenvalue, eigenvector, map, neps, K):
    # Initial point and evolution
    rEps = rfp + epsilon * eigenvector
    rz_path = map(rEps, K=K)

    # Direction of the evolution
    eps_dir = rz_path - rEps 
    norm_eps_dir = np.linalg.norm(eps_dir)
    eps_dir_norm = eps_dir / norm_eps_dir

    # Geometric progression from log_eigenvalue(epsilon) to log_eigenvalue(epsilon + norm_eps_dir)
    eps = np.logspace(
        np.log(epsilon) / np.log(eigenvalue),
        np.log(epsilon + norm_eps_dir) / np.log(eigenvalue),
        neps,
        base=eigenvalue,
    )

    Rs = rfp[0] + eps * eps_dir_norm[0]
    Zs = rfp[1] + eps * eps_dir_norm[1]
    RZs = np.array([[r, z] for r, z in zip(Rs, Zs)])

    return RZs

def fixedpoint(xy, K, m=1):
    xyev = xy
    while m > 0:
        xyev = jitedmap(xyev, K)
        m -= 1

    return xy -xyev

def evolve(startconfig, nintersect, map, K):
        startconfig = np.atleast_2d(startconfig)
        rz_path = np.zeros((2 * startconfig.shape[0], nintersect + 1))
        rz_path[:, 0] = startconfig.flatten()

        for i, rz in enumerate(startconfig):
            ic = rz

            for j in range(nintersect):
                output = map(ic, K=K)
                ic = output
                rz_path[2 * i, j + 1] = output[0]
                rz_path[2 * i + 1, j + 1] = output[1]

        return rz_path

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
    ax.set_title(f'Standard map, K={K:.3f}')
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
    ax.plot(
        out[::2],
        out[1::2],
        '.',
        markersize=msize,
        c='r', zorder=20)
    startconfig = start_config(1e-6, sol.x, lambda_u, -v_u, jitedmap, neps, K)
    path = evolve(startconfig, Nint, jitedmap, K)
    out = path.T.flatten()
    ax.plot(
        out[::2],
        out[1::2],
        '.',
        markersize=msize,
        c='r', zorder=20)
    startconfig = start_config(1e-6, sol.x, lambda_s, v_s, jitedreversed, neps, K)
    path = evolve(startconfig, Nint, jitedreversed, K)
    out = path.T.flatten()
    ax.plot(
        out[::2],
        out[1::2],
        '.',
        markersize=msize,
        c='g', zorder=20)
    startconfig = start_config(1e-6, sol.x, lambda_s, -v_s, jitedreversed, neps, K)
    path = evolve(startconfig, Nint, jitedreversed, K)
    out = path.T.flatten()
    ax.plot(
        out[::2],
        out[1::2],
        '.',
        markersize=msize,
        c='g', zorder=20)
    return fig, ax

if __name__ == "__main__":
    numb = 10
    Ks = np.linspace(0.1, 0.97, numb)
    Nints = np.linspace(50, 17, numb, dtype=int)

    for k, nint in zip(Ks, Nints):
        print(f"Computing for K={k}, Nint={nint}")
        fig, ax = plot(k, nint)
        fig.savefig(f"figs/manifold_{k:.3f}_{nint}.png")
        fig.savefig(f"figs/manifold_{k:.3f}_{nint}.pdf")
        plt.close(fig)