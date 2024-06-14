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

latexplot_folder = Path("../../../../latex/images/plots").absolute()
data_folder = Path("../../../../runs/w7x-gym00-1750/pkl").absolute()
saving_folder = Path("../../w7x").absolute()

sys.path.append(str(latexplot_folder))
from plot_poincare import plot_poincare_simsopt

ratio = 16/9
DPI = 600
file_poincare =  data_folder / "gym00_1750.pkl"

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

# set up the integrator
iparams = dict()
iparams["rtol"] = 1e-12

pparams = dict()
pparams["nrestart"] = 0
pparams["tol"] = 1e-18
pparams['niter'] = 100


fp_x = FixedPoint(ps, pparams, integrator_params=iparams)
fp_o = FixedPoint(ps, pparams, integrator_params=iparams)

fp_x.compute(guess=[5.69956997, 0.52560335], pp=5, qq=4, sbegin=5.2, send=6.2, checkonly=True)
fp_o.compute(guess=[5.6138, 0.8073], pp=5, qq=4, sbegin=5.2, send=6.2, checkonly=True)

results_o = [list(p) for p in zip(fp_o.x, fp_o.y, fp_o.z)]
results_x = [list(p) for p in zip(fp_x.x, fp_x.y, fp_x.z)]

###############################################################################
# Plot and save the poincare plot
###############################################################################

fig, ax = plt.subplots(dpi=DPI)
tys, phi_hits = pickle.load(open(file_poincare, "rb"))
plot_poincare_simsopt(phi_hits, ax)
# ax.set_xlim(5.3, 6.3)
ax.set_xlim(5.1, 6.3)
ax.set_ylim(-1.2, 1.2)

for ii, rr in enumerate(results_o):
    ax.scatter(rr[0], rr[2], marker="o", edgecolors="black", linewidths=1, zorder=20)

    fig.savefig(saving_folder / f"fp_o_{ii}.png", bbox_inches='tight', pad_inches=0.1)

for ii, rr in enumerate(results_x):
    ax.scatter(rr[0], rr[2], marker="X", edgecolors="black", linewidths=1, zorder=20)

    fig.savefig(saving_folder / f"fp_x_{ii}.png", bbox_inches='tight', pad_inches=0.1)