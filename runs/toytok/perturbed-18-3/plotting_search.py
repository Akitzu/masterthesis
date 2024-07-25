from pyoculus.problems import AnalyticCylindricalBfield
from pyoculus.solvers import FixedPoint, Manifold
import matplotlib.pyplot as plt
plt.style.use('lateky')
import numpy as np
from pathlib import Path
from horus import plot_poincare_pyoculus

saving_folder = Path("figs").absolute()

### Creating the pyoculus problem object
print("\nCreating the pyoculus problem object\n")

separatrix = {"type": "circular-current-loop", "amplitude": -10, "R": 6, "Z": -5.5}
maxwellboltzmann = {"m": 18, "n": -3, "d": np.sqrt(2), "type": "maxwell-boltzmann", "amplitude": 0.1}

# Creating the pyoculus problem object, adding the perturbation here use the R, Z provided as center point
pyoproblem = AnalyticCylindricalBfield.without_axis(
    6,
    0,
    0.91,
    0.6,
    perturbations_args=[separatrix],
    Rbegin=1,
    Rend=8,
    niter=800,
    guess=[6.41, -0.7],
    tol=1e-9,
)

# # Adding perturbation after the object is created uses the found axis as center point
pyoproblem.add_perturbation(maxwellboltzmann)

### Finding the X-point
print("\nFinding the X-point\n")

# set up the integrator for the FixedPoint
iparams = dict()
iparams["rtol"] = 1e-12

pparams = dict()
pparams["nrestart"] = 0
pparams["niter"] = 300

# set up the FixedPoint object
fixedpoint = FixedPoint(pyoproblem, pparams, integrator_params=iparams)

# find the X-point
guess = [6.21560891, -4.46981856]
print(f"Initial guess: {guess}")

fixedpoint.compute(guess=guess, pp=0, qq=1, sbegin=4, send=9, tol=1e-10)

if fixedpoint.successful:
    results = [list(p) for p in zip(fixedpoint.x, fixedpoint.y, fixedpoint.z)]
else:
    raise ValueError("X-point not found")

# Manifold
iparams = dict()
iparams["rtol"] = 1e-12

manifold = Manifold(pyoproblem, fixedpoint, fixedpoint, integrator_params=iparams)

# Choose the tangles to work with
manifold.choose(signs=[[1, 1], [1, 1]])

# Finding the clinics
i, s_shift = 6, 2
n_s, n_u = i+s_shift, i-s_shift
manifold.onworking = manifold.inner
# Find the first clinic
eps_s_i, eps_u_i = 2e-5, 2e-5
manifold.find_clinic_single(eps_s_i, eps_u_i, n_s=n_s, n_u=n_u)
# Find the second clinic
manifold.find_clinics(n_points=6)

# Compute the manifold
fig, ax = plt.subplots()
xydata = np.load("data/poincare_18_3.npy")
plot_poincare_pyoculus(xydata, ax, linewidths=0.1)

ax.set_aspect("equal")
ax.scatter(
        pyoproblem._R0, pyoproblem._Z0, marker="o", edgecolors="black", linewidths=1, zorder=11
    )
ax.scatter(
    results[0][0], results[0][2], marker="X", edgecolors="black", linewidths=1, zorder=11
)

eps_s, eps_u = manifold.inner["clinics"][0][1:3]
manifold.compute(nintersect = 9, neps = 300,  eps_s=eps_s, eps_u=eps_u, directions='inner')

# Settings
styledict = {
    "arrowwidth": 5,
    "init_point_size": 15,
    "linewidth": 1,
    "fontsize": 12
}

rfp = manifold.inner["rfp_s"]
lambda_s = manifold.inner["lambda_s"]
lambda_u = manifold.inner["lambda_u"]

vector_s = manifold.inner["vector_s"]
vector_u = manifold.inner["vector_u"]

r_start_s = rfp + eps_s * vector_s
r_start_u = rfp + eps_u * vector_u

r_end_s = manifold.integrate(r_start_s, 1, -1).T[-1]
r_end_u = manifold.integrate(r_start_u, 1, 1).T[-1]

r_s = manifold.start_config(eps_s, rfp, lambda_s, vector_s, neps=10, direction=-1)[1:-1,:].T
r_u = manifold.start_config(eps_u, rfp, lambda_u, vector_u, neps=10, direction=1)[1:-1,:].T

along_s = (np.ones((100,1))*rfp).T + (np.atleast_2d(np.linspace(1e-10, 1e-4, 100)).T * vector_s).T
along_u = (np.ones((100,1))*rfp).T + (np.atleast_2d(np.linspace(1e-10, 1e-4, 100)).T * vector_u).T
evolution = manifold.inner['lfs']

# Plotting
def generate_figure(i, s_shift, lims, ratio=9/16):
    fig, axs = plt.subplots(2, 1)

    # ax = axs[1]
    plot_poincare_pyoculus(xydata, axs[1], linewidths=0.1)
    axs[1].set_aspect('equal')
    axs[1].scatter(*rfp, marker='X', color='tab:orange', edgecolors='black', zorder=10)

    # plot the initial
    axs[1].quiver(*rfp, *vector_s, color='green', alpha=0.4, scale=5)
    axs[1].plot(*along_s, '--', color='grey', alpha=0.5, zorder=-2)
    axs[1].quiver(*rfp, *vector_u, color='red', alpha=0.4, scale=5)
    axs[1].plot(*along_u, '--', color='grey', alpha=0.5, zorder=-2)
    axs[1].set_xlim(lims[0,0])
    axs[1].set_ylim([lims[0,1,0], lims[0,1,0]+ratio*(lims[0,0,1] - lims[0,0,0])])

    # stable
    stable = evolution['stable'][:,1]
    axs[1].plot(stable[::2], stable[1::2], '-', color='green', linewidth=styledict["linewidth"], zorder=12)
    axs[1].scatter(stable[0], stable[1], s=styledict["init_point_size"], marker='s', color='green', zorder=12)
    axs[1].scatter(stable[-2], stable[-1], s=styledict["init_point_size"], marker='o', color='green', zorder=12)
    axs[1].annotate('', xy=stable[-2:], xytext=stable[-4:-2],
                arrowprops=dict(facecolor='green', edgecolor='none', shrink=0.05, width=1, headwidth=styledict["arrowwidth"]),
                zorder=12)

    # unstable
    unstable = evolution['unstable'][:,0]
    axs[1].plot(unstable[::2], unstable[1::2], '-', color='red', linewidth=styledict["linewidth"], zorder=12)
    axs[1].scatter(unstable[0], unstable[1], s=styledict["init_point_size"], marker='s', color='red', zorder=12)
    axs[1].scatter(unstable[-2], unstable[-1], s=styledict["init_point_size"], marker='o', color='red', zorder=12)
    axs[1].annotate('', xy=unstable[-2:], xytext=unstable[-4:-2],
                arrowprops=dict(facecolor='red', edgecolor='none', shrink=0.05, width=1, headwidth=styledict["arrowwidth"]),
                zorder=12)

    # ax = axs[0]
    plot_poincare_pyoculus(xydata, axs[0], linewidths=0.1)

    i_s, i_u = i + s_shift, i - s_shift
    axs[0].set_aspect("equal")
    axs[0].scatter(
            pyoproblem._R0, pyoproblem._Z0, marker="o", edgecolors="black", linewidths=1, zorder=11
        )
    axs[0].scatter(
        results[0][0], results[0][2], marker="X", edgecolors="black", linewidths=1, zorder=11
    )

    # stable
    stable = evolution['stable'][:,i_s]
    axs[0].plot(stable[::2], stable[1::2], '-', color='green', linewidth=styledict["linewidth"], zorder=12)
    axs[0].scatter(stable[0], stable[1], s=styledict["init_point_size"], marker='s', color='green', zorder=12)
    axs[0].scatter(stable[-2], stable[-1], s=styledict["init_point_size"], marker='o', color='green', zorder=12)
    axs[0].annotate('', xy=stable[-2:], xytext=stable[-4:-2],
                arrowprops=dict(facecolor='green', edgecolor='none', shrink=0.05, width=1, headwidth=styledict["arrowwidth"]),
                zorder=12)

    # unstable
    unstable = evolution['unstable'][:,i_u-1]
    axs[0].plot(unstable[::2], unstable[1::2], '-', color='red', linewidth=styledict["linewidth"], zorder=12)
    axs[0].scatter(unstable[0], unstable[1], s=styledict["init_point_size"], marker='s', color='red', zorder=12)
    axs[0].scatter(unstable[-2], unstable[-1], s=styledict["init_point_size"], marker='o', color='red', zorder=12)
    axs[0].annotate('', xy=unstable[-2:], xytext=unstable[-4:-2],
                arrowprops=dict(facecolor='red', edgecolor='none', shrink=0.05, width=1, headwidth=styledict["arrowwidth"]),
                zorder=12)

    # additional settings
    axs[0].set_xlim(lims[1,0])
    axs[0].set_ylim([lims[1,1,0], lims[1,1,0]+lims[1,0,1] - lims[1,0,0]])
    axs[0].set_ylim([lims[1,1,0], lims[1,1,0]+ratio*(lims[1,0,1] - lims[1,0,0])])
    axs[0].set_xlabel("")
    
    return fig, axs

lims = np.array([
                 [[6.202, 6.2055], [-4.4963, -4.4945]], 
                 [[4, 8.2],[0.5, None]],
                ])

ratio = (lims[0,1,1]-lims[0,1,0])/(lims[0,0,1]-lims[0,0,0])

# ratio = (lims[0,1,1]-lims[0,1,0])/(lims[0,0,1]-lims[0,0,0])
fig, axs = generate_figure(6, 0, lims)
# saving

fig.savefig(saving_folder / "search_domain.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
fig.savefig(saving_folder / "search_domain.pdf", dpi=300, bbox_inches='tight', pad_inches=0.1)