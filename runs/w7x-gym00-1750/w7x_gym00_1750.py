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

latexplot_folder = Path("../../latex/images/plots").absolute()
saving_folder = Path("figs").absolute()

sys.path.append(str(latexplot_folder))
from plot_poincare import plot_poincare_simsopt

ratio = 9/16
DPI = 300
file_poincare = "pkl/"
if os.path.exists(file_poincare):
    pass

file_manifold = "pkl/manifold_circlestop.pkl"
if os.path.exists(file_manifold):
    fig, ax = plt.subplots()
    tys, phi_hits = pickle.load(open(file_manifold, "rb"))
    phi_hits = [np.array(phis) for phis in phi_hits]
    plot_poincare_simsopt(phi_hits, ax)

###############################################################################
# Define the W7X cpnfiguration and set up the pyoculus problem
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

###############################################################################
# Plotting clinic evolution
###############################################################################

fig, ax = plt.subplots()
tys, phi_hits = pickle.load(open(file_poincare, "rb"))
phi_hits = [np.array(phis) for phis in phi_hits]
plot_poincare_simsopt(phi_hits, ax)

histories = mp.outer["clinic_history"]
marker = ['d', 's']
colors = ['tab:blue', 'tab:orange']

clinics = mp.outer["clinics"]
ax.scatter(*clinics[0][-1], marker=marker[0], color=colors[0], edgecolor='grey', zorder=13)
ax.scatter(*clinics[1][-1], marker=marker[1], color=colors[1], edgecolor='grey', zorder=13)
ax.set_xlim(5.3, 6)
ax.set_ylim(0.48, ratio*0.7)

fig.set_dpi(DPI)
fig.savefig(saving_folder / f"homoclinic_0.png", bbox_inches='tight', pad_inches=0.1)

for jj in range(8):
    for ii in range(2):
        fb = histories[ii][0][jj]
        ax.scatter(*fb, marker=marker[ii], color=colors[ii], edgecolor='grey', zorder=13)
        fb = histories[ii][1][jj]
        ax.scatter(*fb, marker=marker[ii], color=colors[ii], edgecolor='grey', zorder=13)
        fig.set_dpi(DPI)
        fig.savefig(saving_folder / f"homoclinic_{jj+1}.png", bbox_inches='tight', pad_inches=0.1)

###############################################################################
# Plotting potential evolution
###############################################################################

potentials = mp.outer["potential_integrations"]

fig_potential, ax_potential = plt.subplots()
ax_potential.set_xlim(-0.4, 7.4)
ax_potential.set_ylim(-0.0079, 0.0104)

h1_sum, h2_sum = 0, 0
for ii in range(8):
    h1_sum += potentials[0][0][ii]
    h1_sum -= potentials[0][1][ii]
    h2_sum += potentials[1][0][ii]
    h2_sum -= potentials[1][1][ii]

    ax_potential.scatter(ii, h2_sum-h1_sum, color="tab:blue", zorder=10)
    fig_potential.set_dpi(DPI)
    fig_potential.savefig(saving_folder / f"turnstile_area_{ii}.png", bbox_inches='tight', pad_inches=0.1)

ax_potential.hlines(areas[0,0], -0.4, 7.4, color='grey', linestyle='--', zorder=10)
ax_potential.text(6.36, -0.0055, f"{areas[0,0]:.3e}", va='center')
fig_potential.set_dpi(DPI)
fig_potential.savefig(saving_folder / f"turnstile_area_final.png", bbox_inches='tight', pad_inches=0.1)
