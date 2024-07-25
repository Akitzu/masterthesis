from pyoculus.problems import AnalyticCylindricalBfield
from pyoculus.solvers import FixedPoint, Manifold
import matplotlib.pyplot as plt
plt.style.use('lateky')
import numpy as np
from pathlib import Path
import sys

DPI = 600

current_folder = Path('').absolute()
latexplot_folder = Path("../../../../runs/toytok/perturbed-6-1").absolute()
saving_folder = Path("../../clinic_finder").absolute()
sys.path.append(str(latexplot_folder))
from horus import plot_poincare_pyoculus

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

xydata = np.load(latexplot_folder / "data/perturbed_6_1.npy")
neps = 50
nintersect = 9
iparams = dict()
iparams["rtol"] = 1e-12

manifold = Manifold(pyoproblem, fixedpoint, fixedpoint, integrator_params=iparams)

# Choose the tangles to work with
manifold.choose(signs=[[1, 1], [1, 1]])

print("\nComputing the manifold\n")
eps_s, eps_u = 1e-6, 3e-6
manifold.compute(nintersect = nintersect, neps = neps,  eps_s=eps_s, eps_u=eps_u, directions='inner')

# Settings
styledict = {
    "arrowwidth": 5,
    "init_point_size": 15,
    "linewidth": 1,
    "fontsize": 12
}

textpos = np.array([
                    [[6.20362, -4.496345], [6.20365, -4.49635]],
                    [[6.203333669339457, -4.496002821596208], [6.204044354823329, -4.496099595789755]],
                    [[6.194285533384843, -4.486493076023911], [6.2164034987528085, -4.4887668855477205]],
                    [[6.021202290060593, -4.3029348558889975], [6.465727551955984, -4.33603780092376]],
                    [[5.069787040314116, -3.565288329021239], [7.7375359580630345, -3.5882861645190745]],
                    [[3.7129147459418212, -1.5184809697138801], [8.979419074946152, 0.6893112380783277]],
                    [[7.38, 2.1], [3.65, -0.31]],
                    [[8.795436390963468, 1.2182614545285446], [3.827903923430999, -1.77145716019007]],
                    [[8.26648617451325, -3.1743251255580356], [4.839808685335761, -3.4043034805363903]],
                    [[6.557312346930061, -4.576086823714776], [5.841741166667323, -4.38941608103754]],
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
evolution = manifold.inner['lfs']

# Plotting
def generate_figure(i, s_shift, lims, ratio=9/16):
    fig, axs = plt.subplots(2, 1)

    # ax = axs[1]
    plot_poincare_pyoculus(xydata, axs[1])
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
    stable = evolution['stable'][:,0]
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
    plot_poincare_pyoculus(xydata, axs[0])

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
    axs[0].text(*textpos[i_s][0], f'{i_s}', color='green', fontsize=styledict["fontsize"])
    axs[0].annotate('', xy=stable[-2:], xytext=stable[-4:-2],
                arrowprops=dict(facecolor='green', edgecolor='none', shrink=0.05, width=1, headwidth=styledict["arrowwidth"]),
                zorder=12)

    # unstable
    unstable = evolution['unstable'][:,i_u]
    axs[0].plot(unstable[::2], unstable[1::2], '-', color='red', linewidth=styledict["linewidth"], zorder=12)
    axs[0].scatter(unstable[0], unstable[1], s=styledict["init_point_size"], marker='s', color='red', zorder=12)
    axs[0].scatter(unstable[-2], unstable[-1], s=styledict["init_point_size"], marker='o', color='red', zorder=12)
    axs[0].text(*textpos[i_u][1], f'{i_u}', color='red', fontsize=styledict["fontsize"])
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
                 [[6.2036, 6.203705], [-4.49637, -4.4963]], 
                 [[3, 8],[-1, 2.2]],
                ])
ratio = (lims[0,1,1]-lims[0,1,0])/(lims[0,0,1]-lims[0,0,0])
fig, axs = generate_figure(6, 0, lims, ratio)
# saving
fig.savefig(saving_folder / "clinic_start_0.png", dpi = DPI, bbox_inches='tight', pad_inches=0.1)
plt.close(fig)

# Easier to find there
i, s_shift = 6, 0
n_s, n_u = i+s_shift, i-s_shift
# lims = np.array([
#                  [[6.20361, 6.20366], [-4.49637, -4.49633]], 
#                  [[6, 9],[-4.52, None]],
#                 ])
ratio = (lims[0,1,1]-lims[0,1,0])/(lims[0,0,1]-lims[0,0,0])
fig, axs = generate_figure(i, s_shift, lims, ratio)

# Compute the homoclinic
eps_s_i, eps_u_i = 2e-5, 2e-5
r_guess_s = rfp + eps_s_i * vector_s
r_guess_u = rfp + eps_u_i * vector_u

r_end_s_i = manifold.integrate(r_guess_s, n_s, -1).T[-1]
r_end_u_i = manifold.integrate(r_guess_u, n_u, 1).T[-1]
dr = r_end_s_i - r_end_u_i
dr *= 0.8

axs[0].scatter(*r_end_s_i, marker='d', color='green', edgecolors='black', zorder=13)
axs[0].scatter(*r_end_u_i, marker='d', color='red', edgecolors='black', zorder=13)
axs[0].arrow(*(r_end_u_i+0.05*dr), *dr, head_width=0.3, head_length=0.3, fc='tab:blue', ec='k', zorder=13, linewidth=1, width=0.06)

axs[1].scatter(*r_guess_s, marker='d', color='green', edgecolors='black', zorder=13)
axs[1].scatter(*r_guess_u, marker='d', color='red', edgecolors='black', zorder=13)

fig.savefig(saving_folder / "clinic_start_1.png", dpi = DPI, bbox_inches='tight', pad_inches=0.1)

# Finding the homoclinic
manifold.onworking = manifold.inner
manifold.find_clinic_single(eps_s_i, eps_u_i, n_s=n_s, n_u=n_u)

for ii, hist in enumerate(manifold.inner["history"]):
    if ii%5 == 3:
        fig, axs = generate_figure(i, s_shift, lims, ratio)
        
        r_guess_s = rfp + hist[0,0] * vector_s
        r_guess_u = rfp + hist[0,1] * vector_u
        
        axs[0].scatter(*hist[1], marker='d', color='green', edgecolors='black', zorder=13)
        axs[0].scatter(*hist[2], marker='d', color='red', edgecolors='black', zorder=13)
        
        axs[1].scatter(*r_guess_s, marker='d', color='green', edgecolors='black', zorder=13)
        axs[1].scatter(*r_guess_u, marker='d', color='red', edgecolors='black', zorder=13)
        
        fig.savefig(saving_folder / f"clinic_search_{ii}.png", dpi=DPI, bbox_inches='tight', pad_inches=0.1)

fig, axs = generate_figure(i, s_shift, lims, ratio)
eps_s, eps_u = manifold.inner["clinics"][0][1:3]
r_guess_s = rfp + eps_s * vector_s
r_guess_u = rfp + eps_u * vector_u

hist = manifold.inner["clinics"][0][-2:]
axs[0].scatter(*hist[0], marker='d', color='grey', edgecolors='black', zorder=13)
axs[1].scatter(*r_guess_s, marker='d', color='green', edgecolors='black', zorder=13)
axs[1].scatter(*r_guess_u, marker='d', color='red', edgecolors='black', zorder=13)

fig.savefig(saving_folder / f"clinic_search_end.png", dpi=DPI, bbox_inches='tight', pad_inches=0.1)
breakpoint()


# Find the second clinic
fund = manifold.inner["fundamental_segment"]
guess_i = [fund[0][0]*np.power(manifold.inner["lambda_s"], 1/2), fund[1][0]*np.power(manifold.inner["lambda_u"], 1/2)]
manifold.find_clinic_single(*guess_i, n_s=n_s, n_u=n_u)

### Plotting the final manifold
fig, ax = plt.subplots()
fig, ax = plot_poincare_pyoculus(xydata, ax)
manifold.compute(nintersect = nintersect, neps = 300,  eps_s=eps_s, eps_u=eps_u, directions='inner')
manifold.plot(ax, zorder=12)

marker = ["d", "s"]
colors = ["tab:blue", "tab:orange"]
for i, clinic in enumerate(manifold.inner["clinics"]):
    r_hs, r_hu = clinic[3], clinic[4]
    hs_i = manifold.integrate(r_hs, 6, -1)
    hu_i = manifold.integrate(r_hu, 12, 1)
    ax.scatter(hs_i[0,:], hs_i[1,:], marker=marker[i], color=colors[i], edgecolor='black', zorder=13)
    ax.scatter(hu_i[0,:], hu_i[1,:], marker=marker[i], color=colors[i], edgecolor='black', zorder=13)

fig.set_dpi(DPI)
fig.savefig(saving_folder / "homoclinic_final.png", bbox_inches='tight', pad_inches=0.1)

# contour plot
fig, ax = plt.subplots()
bounds = manifold.find_bounds(eps_s, eps_u)
S, U = np.meshgrid(np.linspace(*bounds[0], 10), np.linspace(*bounds[1], 10))

def residual(s, u):
    rs = rfp + s * vector_s
    ru = rfp + u * vector_u
    rs_e = manifold.integrate_single(rs, n_s, -1, ret_jacobian=False)
    ru_e = manifold.integrate_single(ru, n_u, 1, ret_jacobian=False)
    return rs_e - ru_e

E = np.array([residual(es, eu) for es, eu in zip(S.flatten(), U.flatten())]).reshape(S.shape + (2,))
N = np.linalg.norm(E, axis=2)
ax.contourf(S, U, N, zorder=13)
ax.scatter(manifold.inner["clinics"][0][1], manifold.inner["clinics"][0][2], marker='d', color='tab:blue', edgecolors='black', zorder=13)
ax.scatter(manifold.inner["clinics"][1][1], manifold.inner["clinics"][1][2], marker='s', color='tab:orange', edgecolors='black', zorder=13)

ax.grid('off')
ax.set_xlabel(r"$\varepsilon_s$")
ax.set_ylabel(r"$\varepsilon_u$")

fig.set_dpi(DPI)
fig.savefig(saving_folder / "clinic_search_contour.png", bbox_inches='tight', pad_inches=0.1)