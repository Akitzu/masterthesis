#!/usr/bin/env python

import time
import os
import logging
from pathlib import Path
import numpy as np
from mpi4py import MPI
import simsoptpp as sopp
from simsopt.geo import SurfaceRZFourier, SurfaceClassifier # CurveHelical, CurveXYZFourier, curves_to_vtk
from simsopt.field import BiotSavart
from simsopt.field import (InterpolatedField, SurfaceClassifier, particles_to_vtk,
                           LevelsetStoppingCriterion, load_coils_from_makegrid_file,
                           MinRStoppingCriterion, MaxRStoppingCriterion,
                           MinZStoppingCriterion, MaxZStoppingCriterion,
                           compute_fieldlines
                           )
from simsopt.util import proc0_print, comm_world
from simsopt.field import Current, coils_via_symmetries
from simsopt.configs import get_w7x_data
from pyoculus.problems import SimsoptBfieldProblem
from pyoculus.solvers import FixedPoint, Manifold
import matplotlib.pyplot as plt
from horus import poincare
import pickle
import sys

latexplot_folder = Path("../../latex/images/plots").absolute()
saving_folder = Path("figs").absolute()

sys.path.append(str(latexplot_folder))
from plot_poincare import plot_poincare_simsopt

ratio = 16/9
DPI = 300
file_poincare = "pkl/gym00_1750.pkl"
file_manifold_stable = "pkl/manifold_stable.pkl"
file_manifold_unstable = "pkl/manifold_unstable.pkl"

###############################################################################
# Define the W7X configuration and set up the pyoculus problem
###############################################################################

nfp = 5  # Number of field periods
curves, currents, ma = get_w7x_data()

# GYM00+1750 currents
currents_gym = [Current(1.109484) * 1e6 for _ in range(5)]
currents_gym.append(Current(-0.3661) * 1e6)
currents_gym.append(Current(-0.3661) * 1e6)

coils = coils_via_symmetries(curves, currents_gym, 5, True)

# Surface delimiter
# from pyoculus.problems import surf_from_coils
# surf = surf_from_coils(coils, ncoils=7, mpol=5, ntor=5)

surf = SurfaceRZFourier.from_nphi_ntheta(mpol=5, ntor=5, stellsym=True, nfp=5, range="full torus", nphi=64, ntheta=24)
surf.fit_to_curve(ma, 1.5, flip_theta=False)

surfclassifier = SurfaceClassifier(surf, h=0.1, p=1)

# Setting the problem
R0, _, Z0 = ma.gamma()[0,:]
bs = BiotSavart(coils)
ps = SimsoptBfieldProblem.without_axis([5.98, 0], nfp, bs)
R0, Z0 = ps._R0, ps._Z0
# ps = SimsoptBfieldProblem(R0=R0, Z0=Z0, Nfp=nfp, mf=bs)
ps = SimsoptBfieldProblem(R0=R0, Z0=Z0, Nfp=nfp, mf=bs, interpolate=True, surf=surf)

################################################################################
## Find fixed point and set the manifold
################################################################################

fig, ax = plt.subplots()
if os.path.exists(file_poincare):
    tys, phi_hits = pickle.load(open(file_poincare, "rb"))
    plot_poincare_simsopt(phi_hits, ax)
    tys, phi_hits = pickle.load(open(file_manifold_stable, "rb"))
    plot_poincare_simsopt(phi_hits, ax, color='green')
    tys, phi_hits = pickle.load(open(file_manifold_unstable, "rb"))
    plot_poincare_simsopt(phi_hits, ax, color='red')

# set up the integrator
iparams = dict()
iparams["rtol"] = 1e-12

pparams = dict()
pparams["nrestart"] = 0
pparams["tol"] = 1e-18
pparams['niter'] = 100


fp_x1 = FixedPoint(ps, pparams, integrator_params=iparams)
fp_x2 = FixedPoint(ps, pparams, integrator_params=iparams)

fp_x1.compute(guess=[5.69956997, 0.52560335], pp=5, qq=4, sbegin=5.2, send=6.2, checkonly=True)
fp_x2.compute(guess=[5.883462104879646, 0.6556749703570318], pp=5, qq=4, sbegin=5.2, send=6.2, checkonly=True)

results_x = [list(p) for p in zip(fp_x1.x, fp_x1.y, fp_x1.z)]
for rr in results_x:
    ax.scatter(rr[0], rr[2], marker="X", edgecolors="black", linewidths=1, zorder=20)
fig.savefig(saving_folder / "gym99_1750_fixedpoints.png", dpi=DPI, bbox_inches='tight', pad_inches=0.1)

# Working on manifold
iparam = dict()
iparam["rtol"] = 1e-13

mp = Manifold(ps, fp_x1, fp_x2, integrator_params=iparam)
mp.choose(signs=[[1, -1], [-1, 1]], order=False)

###############################################################################
# Computation of the turnstile area
###############################################################################

proc0_print("Finding the heteroclinic points")

mp.onworking = mp.outer
mp.find_clinic_single(0.0005262477692494645, 0.0011166626256106193, n_s=7, n_u=7)
mp.find_clinic_single(0.0006491052045732048, 0.0013976539980315229, n_s=7, n_u=6)

mp.turnstile_area()
areas = mp.outer["areas"]

mp.onworking = mp.inner
mp.find_clinic_single(0.0009232307475864197, 0.0009969436196850716, method="lm")
mp.find_clinic_single(0.0011721658811519728, 0.0020577144928870914, n_s=6, n_u=5)

mp.turnstile_area()

B_phi_0 = ps.B([ps._R0, 0., ps._Z0])[1] * ps._R0

np.save("area_outer.npy", mp.outer["areas"])
np.save("area_inner.npy", mp.inner["areas"])

# import pandas as pd
# data = [
#     {"type": "inner", "area": inner_areas[inner_areas > 0].sum(), "Error_by_diff": manifold.inner["areas"][:, 1][inner_areas > 0].sum(), "Error_by_estim": manifold.inner["areas"][:, 2][inner_areas > 0].sum(), "total_sum": inner_areas.sum()},
#     {"type": "outer", "area": outer_areas[outer_areas > 0].sum(), "Error_by_diff": manifold.outer["areas"][:, 1][outer_areas > 0].sum(), "Error_by_estim": manifold.outer["areas"][:, 2][outer_areas > 0].sum(), "total_sum": outer_areas.sum()},
# ]

# df = pd.DataFrame(data)
# df.to_csv("areas.csv")

###############################################################################
# Plotting clinic evolution
###############################################################################

histories = mp.outer["clinic_history"]
marker = ['d', 's']
colors = ['royalblue', 'royalblue']

clinics = mp.outer["clinics"]
ax.scatter(*clinics[0][-1], marker=marker[0], color=colors[0], edgecolor='cyan', zorder=13)
ax.scatter(*clinics[1][-1], marker=marker[1], color=colors[1], edgecolor='cyan', zorder=13)
ax.set_xlim(5.3, 6)
ax.set_ylim(0.48, ratio*0.7)

fig.set_dpi(DPI)
fig.savefig(saving_folder / f"homoclinic_0.png", bbox_inches='tight', pad_inches=0.1)

for jj in range(8):
    for ii in range(2):
        fb = histories[ii][0][jj]
        ax.scatter(*fb, marker=marker[ii], color=colors[ii], edgecolor='cyan', zorder=13)
        fb = histories[ii][1][jj]
        ax.scatter(*fb, marker=marker[ii], color=colors[ii], edgecolor='cyan', zorder=13)
        fig.set_dpi(DPI)
        fig.savefig(saving_folder / f"homoclinic_{jj+1}.png", bbox_inches='tight', pad_inches=0.1)

###############################################################################
# Plotting potential evolution
###############################################################################

potentials = mp.outer["potential_integrations"]

fig_potential, ax_potential = plt.subplots()
ax_potential.set_xlim(0.6, 8.4)
ax_potential.set_ylim(-0.0079, 0.0104)
ax_potential.set_xlabel(r"Integration", fontsize=16)
ax_potential.set_ylabel(r"Turnstile flux", fontsize=16)

h1_sum, h2_sum = 0, 0
sum_array = np.empty(8)
for ii in range(8):
    h1_sum += potentials[0][0][ii]
    h1_sum -= potentials[0][1][ii]
    h2_sum += potentials[1][0][ii]
    h2_sum -= potentials[1][1][ii]

    sum_array[ii] = h2_sum-h1_sum
    ax_potential.scatter(ii+1, h2_sum-h1_sum, color="tab:blue", zorder=10)
    fig_potential.set_dpi(DPI)
    fig_potential.savefig(saving_folder / f"turnstile_area_{ii}.png", bbox_inches='tight', pad_inches=0.1)

ax_potential.hlines(areas[0,0], 0.6, 8.4, color='grey', linestyle='--', zorder=10)
ax_potential.text(7.36, -0.0055, f"{areas[0,0]:.3e}", va='center')
fig_potential.set_dpi(DPI)
fig_potential.savefig(saving_folder / f"turnstile_area_final.png", bbox_inches='tight', pad_inches=0.1)

fig_log, ax_log = plt.subplots()
# ax_log.semilogy(1+np.arange(len(sum_array[1:])), np.abs(sum_array[1:]), marker='o', color='tab:blue')
ax_log.semilogy(1+np.arange(len(sum_array[1:-1])), np.abs(sum_array[1:-1]-sum_array[-1]), marker='o', color='tab:blue')
ax_log.set_xlabel(r"Iteration", fontsize=16)
# ax_log.set_ylabel(r"Action difference $\sum_t \lambda(h^t_2)-\lambda(h^t_1)$", fontsize=16)
ax_log.set_ylabel(r"Action difference $\sum_t \lambda(h^t_2)-\lambda(h^t_1)$ - $end_{area}$", fontsize=16)
fig_log.savefig(saving_folder / f"potlog_2.png", bbox_inches='tight', pad_inches=0.1, dpi=DPI)


###############################################################################
# Save initial points for manifold tracing
###############################################################################

# rfp_s = mp.outer["rfp_s"]
# rfp_u = mp.outer["rfp_u"]
# lambda_s = mp.outer["lambda_s"]
# lambda_u = mp.outer["lambda_u"]
# vector_s = mp.outer["vector_s"]
# vector_u = mp.outer["vector_u"]

# fund = mp.outer["fundamental_segment"]
# eps_s_1, eps_u_1 = fund[0][0], fund[1][0]
# eps_s_2, eps_u_2 = mp.outer["clinics"][1][1:3]
# eps_s_3, eps_u_3 = fund[0][1], fund[1][1]

# neps = 2*25+1
# start_eps_s = np.concatenate((
#         np.logspace(
#             np.log(eps_s_1) / np.log(lambda_s),
#             np.log(eps_s_2) / np.log(lambda_s),
#             int(neps/2),
#             base=lambda_s,
#             endpoint=False
#         ),
#         np.logspace(
#             np.log(eps_s_2) / np.log(lambda_s),
#             np.log(eps_s_3) / np.log(lambda_s),
#             int(neps/2)+1,
#             base=lambda_s,
#         )
#     ))
# start_eps_u = np.concatenate((
#         np.logspace(
#             np.log(eps_u_1) / np.log(lambda_u),
#             np.log(eps_u_2) / np.log(lambda_u),
#             int(neps/2),
#             base=lambda_u,
#             endpoint=False
#         ),
#         np.logspace(
#             np.log(eps_u_2) / np.log(lambda_u),
#             np.log(eps_u_3) / np.log(lambda_u),
#             int(neps/2)+1,
#             base=lambda_u,
#         )
#     ))

# rz_start_s = (np.ones((neps,1))*rfp_s) + (np.atleast_2d(start_eps_s).T * vector_s)
# rz_start_u = (np.ones((neps,1))*rfp_u) + (np.atleast_2d(start_eps_u).T * vector_u)

# np.save("pkl/rz_start_s.npy", rz_start_s)
# np.save("pkl/rz_start_u.npy", rz_start_u)