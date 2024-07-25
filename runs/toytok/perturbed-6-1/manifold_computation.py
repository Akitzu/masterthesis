from pyoculus.problems import AnalyticCylindricalBfield
from pyoculus.solvers import FixedPoint, Manifold
import matplotlib.pyplot as plt
plt.style.use('lateky')
import numpy as np
from pathlib import Path
from horus import plot_poincare_pyoculus
from matplotlib import patheffects

DPI = 600
# ratio = (6+2.5)/(9.2-3.5)
saving_folder = Path("figs").absolute()

### Creating the pyoculus problem object
print("\nCreating the pyoculus problem object\n")

separatrix = {"type": "circular-current-loop", "amplitude": -10, "R": 6, "Z": -5.5}
maxwellboltzmann = {"m": 6, "n": -1, "d": np.sqrt(2), "type": "maxwell-boltzmann", "amplitude": 1e-1}

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

xydata = np.load("data/perturbed_6_1.npy")
neps = 50
nintersect = 9
iparams = dict()
iparams["rtol"] = 1e-12

manifold = Manifold(pyoproblem, fixedpoint, fixedpoint, integrator_params=iparams)

# Choose the tangles to work with
manifold.choose(signs=[[1, 1], [1, 1]])

print("\nComputing the manifold\n")
eps_s, eps_u = 1e-6, 1e-6
manifold.compute(nintersect = nintersect, neps = neps,  eps_s=eps_s, eps_u=eps_u, directions='inner')

# Settings
styledict = {
    "arrowwidth": 13,
    "init_point_size": 15,
    "linewidth": 1,
    "fontsize": 12
}


textpos = np.array([
                    [[6.20362, -4.496345], [6.20365, -4.49635]],
                    [[6.203333669339457, -4.496002821596208], [6.204044354823329, -4.496099595789755]],
                    [[6.194285533384843, -4.486493076023911], [6.2164034987528085, -4.4887668855477205]],
                    [[6.021202290060593, -4.3029348558889975], [6.465727551955984, -4.33603780092376]],
                    [[4.69, -3.26], [8.3, -3.24]],
                    [[3.6, -0.64], [8.28, 1.62]],
                    [[4.72, 2.08], [4, 1.32]],
                    [[8.95, -1.90], [3.89, -1.95]],
                    [[7.48, -4.04], [5.13, -3.93]],
                    [[6.88, -4.98], [5.36, -4.75]],
                ])
lims = np.array([
                 [[6.203615, 6.203657], [-4.496363, -4.496333]], 
                #  [[6.2030, 6.2045],[-4.4965, -4.4955]],
                #  [[6.18, 6.23],[-4.5056, -4.4674]],
                #  [[5.7, 6.8],[-4.7, -3.8]],
                #  [[3.5, 9.2],[-6, 2.5]],
                #  [[3.5, 9.2],[-6, 2.5]],
                #  [[3.5, 9.2],[-6, 2.5]],
                #  [[3.5, 9.2],[-6, 2.5]],
                #  [[3.5, 9.2],[-6, 2.5]],
                #  [[5.4, 7],[-6, -3.7]],
                ])

## Plotting the starting point
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

fig, ax = plt.subplots()
# fig, ax = plot_poincare_pyoculus(xydata, ax)
ax.set_aspect('equal')
ax.set_xlabel(r"$R$", fontsize=14)
ax.set_ylabel(r"$Z$", fontsize=14)
ax.scatter(*rfp, marker='X', color='tab:orange', edgecolors='black', zorder=10)

# plot the initial
ax.quiver(*rfp, *vector_s, color='green', alpha=0.4, scale=5)
ax.plot(*along_s, '--', color='grey', alpha=0.5, zorder=-2)
ax.quiver(*rfp, *vector_u, color='red', alpha=0.4, scale=5)
ax.plot(*along_u, '--', color='grey', alpha=0.5, zorder=-2)
ax.set_xlim(lims[0][0])
ax.set_ylim(lims[0][1])
# ax.set_ylim(lims[0][1][0], ratio*(lims[0][0][1]-lims[0][0][0])+lims[0][1][0])
# fig.savefig(saving_folder / "manifold_start_0.png", bbox_inches='tight', pad_inches=0.1)

# add point
ax.scatter(*r_end_s, marker='o', color='green', edgecolors='black', zorder=10)
ax.scatter(*r_start_u, marker='o', color='red', edgecolors='black', zorder=10)
# fig.savefig(saving_folder / "manifold_start_1.png", bbox_inches='tight', pad_inches=0.1)

# add evolution
ax.scatter(*r_start_s, marker='>', color='green', edgecolors='black', zorder=10)
ax.scatter(*r_end_u, marker='>', color='red', edgecolors='black', zorder=10)
# fig.savefig(saving_folder / "manifold_start_2.png", bbox_inches='tight', pad_inches=0.1)

# add starting config
ax.scatter(*r_s, s=10, marker='o', color='green', edgecolors='black', zorder=9)
ax.scatter(*r_u, s=10, marker='o', color='red', edgecolors='black', zorder=9)
fig.savefig(saving_folder / "manifold_start.png", dpi = DPI, bbox_inches='tight', pad_inches=0.1)
fig.savefig(saving_folder / "manifold_start.pdf", dpi = DPI, bbox_inches='tight', pad_inches=0.1)
plt.close(fig)
## Plotting the manifold computations
evolution = manifold.inner['lfs']

fig, ax = plt.subplots()
fig, ax = plot_poincare_pyoculus(xydata, ax, linewidths=0.1, s=0.5)
ax.set_aspect("equal")
ax.scatter(
        pyoproblem._R0, pyoproblem._Z0, marker="o", edgecolors="black", linewidths=1, zorder=11
    )
ax.scatter(
    results[0][0], results[0][2], marker="X", edgecolors="black", linewidths=1, zorder=11
)

for i in range(nintersect+1):

    # stable
    stable = evolution['stable'][:,i]
    ax.plot(stable[::2], stable[1::2], '.-', color='green', linewidth=styledict["linewidth"], zorder=12)
    ax.scatter(stable[0], stable[1], s=styledict["init_point_size"], marker='s', color='green', linewidth=3, zorder=12)
    if i > 3:
        ax.text(*textpos[i][0], f'{i}', color='green', fontsize=styledict["fontsize"],
                path_effects=[patheffects.withStroke(linewidth=1, foreground="black")])
        ax.annotate('', xy=stable[-2:], xytext=stable[-4:-2],
                    arrowprops=dict(facecolor='green', edgecolor='none', shrink=0.05, width=1, headwidth=styledict["arrowwidth"]),
                    zorder=12)

    # unstable
    unstable = evolution['unstable'][:,i]
    ax.plot(unstable[::2], unstable[1::2], '.-', color='red', linewidth=styledict["linewidth"], zorder=12)
    ax.scatter(unstable[0], unstable[1], s=styledict["init_point_size"], marker='s', color='red', linewidth=3, zorder=12)
    if i > 3:
        ax.text(*textpos[i][1], f'{i}', color='red', fontsize=styledict["fontsize"],
                path_effects=[patheffects.withStroke(linewidth=1, foreground="black")])
        ax.annotate('', xy=unstable[-2:], xytext=unstable[-4:-2],
                    arrowprops=dict(facecolor='red', edgecolor='none', shrink=0.05, width=1, headwidth=styledict["arrowwidth"]),
                    zorder=12)

    # additional settings
    # ax.set_xlim(lims[i][0])
    # ax.set_ylim(lims[i][1][0], ratio*(lims[i][0][1]-lims[i][0][0])+lims[i][1][0])
    # fig.set_dpi(DPI)

    # Save the figure
    # fig.savefig(saving_folder / "manifold_{i}.pdf")
    # fig.savefig(saving_folder / f"manifold_{i}.png", bbox_inches='tight', pad_inches=0.1)
    # plt.close(fig)
fig.savefig(saving_folder / "manifold.pdf", dpi = DPI, bbox_inches='tight', pad_inches=0.1)
fig.savefig(saving_folder / "manifold.png", dpi = DPI, bbox_inches='tight', pad_inches=0.1)

### Plotting the final manifold
fig, ax = plt.subplots()
fig, ax = plot_poincare_pyoculus(xydata, ax, linewidths=0.1, s=0.5)
ax.set_aspect("equal")
ax.scatter(
        pyoproblem._R0, pyoproblem._Z0, marker="o", edgecolors="black", linewidths=1, zorder=11
    )
ax.scatter(
    results[0][0], results[0][2], marker="X", edgecolors="black", linewidths=1, zorder=11
)

indices = [3, 8, 9]
textpos = np.array([
                    [[6.20362, -4.496345], [6.20365, -4.49635]],
                    [[6.203333669339457, -4.496002821596208], [6.204044354823329, -4.496099595789755]],
                    [[6.194285533384843, -4.486493076023911], [6.2164034987528085, -4.4887668855477205]],
                    [[5.808, -4.341], [6.526, -4.415]],
                    [[5.069787040314116, -3.565288329021239], [7.7375359580630345, -3.5882861645190745]],
                    [[3.7129147459418212, -1.5184809697138801], [8.979419074946152, 0.6893112380783277]],
                    [[3.6899169104439857, 1.1032722770393661], [7.55355327408035, 2.3221575584246477]],
                    [[8.795436390963468, 1.2182614545285446], [3.827903923430999, -1.77145716019007]],
                    [[6.934, -4.219], [5.539, -3.958]],
                    [[6.657, -5.378], [5.702, -4.701]],
                ])

for i in range(nintersect+1):

    # stable
    stable = evolution['stable'][:,i]
    ax.plot(stable[::2], stable[1::2], '.-', color='green', linewidth=styledict["linewidth"], zorder=12)
    ax.scatter(stable[0], stable[1], s=styledict["init_point_size"], marker='s', color='green', linewidth=3, zorder=12)
    if i in indices:
        ax.text(*textpos[i][0], f'{i}', color='green', fontsize=styledict["fontsize"],
                path_effects=[patheffects.withStroke(linewidth=1, foreground="black")])
        ax.annotate('', xy=stable[-2:], xytext=stable[-4:-2],
                    arrowprops=dict(facecolor='green', edgecolor='none', shrink=0.05, width=1, headwidth=styledict["arrowwidth"]),
                    zorder=12)

    # unstable
    unstable = evolution['unstable'][:,i]
    ax.plot(unstable[::2], unstable[1::2], '.-', color='red', linewidth=styledict["linewidth"], zorder=12)
    ax.scatter(unstable[0], unstable[1], s=styledict["init_point_size"], marker='s', color='red', linewidth=3, zorder=12)
    if i in indices:
        ax.text(*textpos[i][1], f'{i}', color='red', fontsize=styledict["fontsize"],
                path_effects=[patheffects.withStroke(linewidth=1, foreground="black")])
        ax.annotate('', xy=unstable[-2:], xytext=unstable[-4:-2],
                    arrowprops=dict(facecolor='red', edgecolor='none', shrink=0.05, width=1, headwidth=styledict["arrowwidth"]),
                    zorder=12)
        
ax.set_xlim(4.7, 8)
ax.set_ylim(-6, -3)

fig.savefig(saving_folder / "manifold_closer.pdf", dpi = DPI, bbox_inches='tight', pad_inches=0.1)
fig.savefig(saving_folder / "manifold_closer.png", dpi = DPI, bbox_inches='tight', pad_inches=0.1)