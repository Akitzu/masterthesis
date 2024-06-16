from pyoculus.problems import AnalyticCylindricalBfield
from pyoculus.solvers import FixedPoint, Manifold
import matplotlib.pyplot as plt
plt.style.use('lateky')
import numpy as np
from pathlib import Path
import sys

DPI = 600

current_folder = Path('').absolute()
latexplot_folder = Path("../../../../latex/images/plots").absolute()
saving_folder = Path("../../turnstile_calc").absolute()
sys.path.append(str(latexplot_folder))
from plot_poincare import plot_poincare_pyoculus

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

xydata = np.load(latexplot_folder / "toytok-6-1/poincare.npy")

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
manifold.find_clinics(n_points=2)

print("\nComputing the manifold\n")

## Plotting the starting point
rfp = manifold.inner["rfp_s"]
lambda_s = manifold.inner["lambda_s"]
lambda_u = manifold.inner["lambda_u"]
vector_s = manifold.inner["vector_s"]
vector_u = manifold.inner["vector_u"]

fund = manifold.inner["fundamental_segment"]
eps_s_1, eps_u_1 = fund[0][0], fund[1][0]
eps_s_2, eps_u_2 = manifold.inner["clinics"][1][1:3]
eps_s_3, eps_u_3 = fund[0][1], fund[1][1]

neps = 2*25+1
start_eps_s = np.concatenate((
        np.logspace(
            np.log(eps_s_1) / np.log(lambda_s),
            np.log(eps_s_2) / np.log(lambda_s),
            int(neps/2),
            base=lambda_s,
            endpoint=False
        ),
        np.logspace(
            np.log(eps_s_2) / np.log(lambda_s),
            np.log(eps_s_3) / np.log(lambda_s),
            int(neps/2)+1,
            base=lambda_s,
        )
    ))
start_eps_u = np.concatenate((
        np.logspace(
            np.log(eps_u_1) / np.log(lambda_u),
            np.log(eps_u_2) / np.log(lambda_u),
            int(neps/2),
            base=lambda_u,
            endpoint=False
        ),
        np.logspace(
            np.log(eps_u_2) / np.log(lambda_u),
            np.log(eps_u_3) / np.log(lambda_u),
            int(neps/2)+1,
            base=lambda_u,
        )
    ))

rz_start_s = (np.ones((neps,1))*rfp) + (np.atleast_2d(start_eps_s).T * vector_s)
rz_start_u = (np.ones((neps,1))*rfp) + (np.atleast_2d(start_eps_u).T * vector_u)

# computing the manifold
nintersect = 9
stable_path = manifold.integrate(rz_start_s, nintersect=nintersect, direction=-1)
unstable_path = manifold.integrate(rz_start_u, nintersect=nintersect, direction=1)

# Plotting the manifold computations
def plot_contour(fwd_ev, ax, num = 2, onlyclinics=False):
    i_s, i_u = 8-fwd_ev, 3+fwd_ev
    stable = stable_path[:,i_s]
    unstable = unstable_path[:,i_u]
    stable_2, stable_1 = stable[:neps+1], stable[neps-1:]
    unstable_1, unstable_2 = unstable[:neps+1], unstable[neps-1:]

#     if num == 2:
#         ax.scatter(*unstable_2[-2:], marker='d', color='tab:blue', edgecolor='grey', zorder=13)
#         ax.scatter(*unstable_2[:2], marker='s', color='tab:orange', edgecolor='grey', zorder=13)
#         if not onlyclinics:
#             ax.plot(unstable_2[::2], unstable_2[1::2], '.-', color='red', linewidth=2, markersize=1, zorder=12)
#             ax.plot(stable_2[::2], stable_2[1::2], '.-', color='green', linewidth=2, markersize=1, zorder=12)
#     if num == 1:
#         ax.scatter(*unstable_1[:2], marker='d', color='tab:blue', edgecolor='grey', zorder=13)
#         ax.scatter(*unstable_1[-2:], marker='s', color='tab:orange', edgecolor='grey', zorder=13)
#         if not onlyclinics:
#             ax.plot(unstable_1[::2], unstable_1[1::2], '.-', color='red', linewidth=2, markersize=1, zorder=12)
#             ax.plot(stable_1[::2], stable_1[1::2], '.-', color='green', linewidth=2, markersize=1, zorder=12)

# # Plotting the contour evolution
# ratio = 16/9

# for ff in range(1,7):
#     fig, ax = plt.subplots()
#     ax.set_aspect('equal')
#     plot_contour(0, ax)
#     plot_contour(ff, ax)
#     cxlim, cylim = ax.get_xlim(), ax.get_ylim()
#     plot_poincare_pyoculus(xydata, ax)
#     ax.set_xlim(cxlim)
#     ax.set_ylim(cylim[0], cylim[0]+ratio*(cxlim[1]-cxlim[0]))
#     fig.set_dpi(DPI)
#     fig.savefig(saving_folder / f"coutour_fwd_{ff}.png", bbox_inches='tight', pad_inches=0.1)
#     plt.close(fig)

# for bb in range(0,2):
#     fig, ax = plt.subplots()
#     plot_contour(-bb, ax)
#     plot_contour(6, ax)
#     cxlim, cylim = ax.get_xlim(), ax.get_ylim()
#     plot_poincare_pyoculus(xydata, ax)
#     ax.set_xlim(cxlim)
#     ax.set_ylim(cylim[0], cylim[0]+ratio*(cxlim[1]-cxlim[0]))
#     fig.set_dpi(DPI)
#     fig.savefig(saving_folder / f"coutour_bwd_{bb}.png", bbox_inches='tight', pad_inches=0.1)
#     plt.close(fig)

##############################
# Plotting the convergence
##############################

manifold.turnstile_area(True)
# histories = manifold.inner["clinic_history"]
# potentials = manifold.inner["potential_integrations"]

# fig, ax = plt.subplots()
# plot_poincare_pyoculus(xydata, ax)
# plot_contour(0, ax, onlyclinics=True)
# fig.set_dpi(DPI)
# fig.savefig(saving_folder / "turnstile_poincare_0.png", bbox_inches='tight', pad_inches=0.1)

# fig_potential, ax_potential = plt.subplots()
# ax_potential.set_xlim(-0.4, 8.4)
# ax_potential.set_ylim(-17, 7)
# ax_potential.set_xlabel(r"Integration", fontsize=16)
# # ax_potential.set_ylabel(r"$\sum_t \lambda(h^t_2)-\lambda(h^t_1)$", fontsize=16)
# ax_potential.set_ylabel(r"Turnstile flux", fontsize=16)

# h1_sum, h2_sum = 0, 0
# for ii, (h1, h2) in enumerate(zip(potentials[0][0], potentials[1][0])):
#     if ii > 3:
#         break
#     h1_sum += h1
#     h2_sum += h2
#     ax_potential.scatter(ii, h2_sum-h1_sum, color="tab:blue", zorder=10)
#     fig_potential.set_dpi(DPI)
#     fig_potential.savefig(saving_folder / f"turnstile_area_{ii}.png", bbox_inches='tight', pad_inches=0.1)

# sum_array = np.empty(5)
# for ii in range(5):
#     h1_sum += potentials[0][0][ii+4]
#     h1_sum -= potentials[0][1][ii]

#     h2_sum += potentials[1][0][ii+4]
#     h2_sum -= potentials[1][1][ii]

#     sum_array[ii] = h2_sum-h1_sum
#     ax_potential.scatter(ii+4, h2_sum-h1_sum, color="tab:blue", zorder=10)
#     fig_potential.set_dpi(DPI)
#     fig_potential.savefig(saving_folder / f"turnstile_area_{ii+4}.png", bbox_inches='tight', pad_inches=0.1)

# areas = manifold.inner["areas"]
# ax_potential.hlines(areas[0,0], -0.4, 8.4, color='grey', linestyle='--', zorder=10)
# ax_potential.text(7.4, 0.2, f"{areas[0,0]:.3e}", va='center')
# fig_potential.set_dpi(DPI)
# fig_potential.savefig(saving_folder / f"turnstile_area_final.png", bbox_inches='tight', pad_inches=0.1)

# # fig_log, ax_log = plt.subplots()
# # ax_log.semilogy(np.arange(5), np.abs(sum_array), 'o-', color='tab:blue', zorder=10)

# # # plot of the last 5 step as convergence
# # fig_pot_conv, ax_pot_conv = plt.subplots()
# # ax_pot_conv

# # Plotting the clinic evolution
# marker = ['d', 's']
# colors = ['tab:blue', 'tab:orange']
# for jj in range(4):
#     for ii in range(2):
#         fb = histories[ii][0][jj]
#         ax.scatter(*fb, marker=marker[ii], color=colors[ii], edgecolor='grey', zorder=13)
#     fig.set_dpi(DPI)
#     fig.savefig(saving_folder / f"turnstile_poincare_{jj}.png", bbox_inches='tight', pad_inches=0.1)

# for jj in range(5):
#     for ii in range(2):
#         fb = histories[ii][0][jj+4]
#         ax.scatter(*fb, marker=marker[ii], color=colors[ii], edgecolor='grey', zorder=13)
#         fb = histories[ii][1][jj]
#         ax.scatter(*fb, marker=marker[ii], color=colors[ii], edgecolor='grey', zorder=13)
#     if jj == 0:
#         ax.set_xlim(5.5, 6.8)
#         ax.set_ylim(-4.8, -4.8+ratio*1.3)
#     else:
#         ax.set_xlim(6, 6.4)
#         ax.set_ylim(-4.51, -4.51+ratio*0.4)

#     fig.set_dpi(DPI)
#     fig.savefig(saving_folder / f"turnstile_poincare_{jj+4}.png", bbox_inches='tight', pad_inches=0.1)

##############################
# Verification
##############################

# loop
stable = stable_path[:,8]
unstable = unstable_path[:,3]
stable_2 = stable[:neps+1].reshape(-1,2)
unstable_2 = unstable[neps-1:].reshape(-1,2)
loop = np.concatenate((stable_2[::-2], unstable_2[::-1]))

# Calculation of the area using the shoelace formula
x = loop[:, 0]
y = loop[:, 1]
area_shoe = 0.5*np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

# Plotting verification 1
fig, ax = plt.subplots()
plot_contour(0, ax)
cxlim, cylim = ax.get_xlim(), ax.get_ylim()
plot_poincare_pyoculus(xydata, ax)
ax.set_xlim(cxlim)
ax.set_ylim(cylim)
fig.set_dpi(DPI)
# fig.savefig(saving_folder / "verification_0.png", bbox_inches='tight', pad_inches=0.1)
ax.plot(loop[:,0], loop[:,1], '.:', color='black', linewidth=2, markersize=2, zorder=12)
ax.scatter(6.7913, -4.0464, color='black', label=r'$B^\phi$ evaluation point', zorder=14)
ax.legend(loc='lower right')
fig.set_dpi(DPI)
# fig.savefig(saving_folder / "verification_1.png", bbox_inches='tight', pad_inches=0.1)

# Calculation of the area using triangle approximation
A = unstable_2[0]
B = unstable_2[-1]
C = np.array([6.7833, -4.111])
area_tri = 0.5*np.abs(np.cross(B-A, C-A))

fig, ax = plt.subplots()
plot_contour(0, ax)
cxlim, cylim = ax.get_xlim(), ax.get_ylim()
plot_poincare_pyoculus(xydata, ax)
ax.set_xlim(cxlim)
ax.set_ylim(cylim)
ax.plot([A[0], B[0], C[0], A[0]], [A[1], B[1], C[1], A[1]], 'o-', color='black', linewidth=2, markersize=2, zorder=14)
ax.scatter(6.7913, -4.0464, color='black', label=r'$B^\phi$ evaluation point', zorder=14)
ax.legend(loc='lower right')
fig.set_dpi(DPI)
# fig.savefig(saving_folder / "verification_2.png", bbox_inches='tight', pad_inches=0.1)

# B^phi estimation
B_phi = pyoproblem.B([6.7913, 0., -4.0464])[1] * 6.7913
B_phi_0 = pyoproblem.B([pyoproblem._R0, 0., pyoproblem._Z0])[1] * pyoproblem._R0

print(f"Area using the shoelace formula: {B_phi*area_shoe:.3e}")
print(f"Area using the triangle approximation: {B_phi*area_tri:.3e}")