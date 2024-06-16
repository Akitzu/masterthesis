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
from simsopt.util import proc0_print
from simsopt.field import Current, coils_via_symmetries
from simsopt.configs import get_ncsx_data
from pyoculus.problems import SimsoptBfieldProblem
from pyoculus.solvers import FixedPoint, Manifold
import matplotlib.pyplot as plt
from horus import poincare
import pickle
import sys

###############################################################################
# Define the NCSX cpnfiguration and set up the pyoculus problem
###############################################################################

nfp = 3 # Number of field periods
curves, currents, ma = get_ncsx_data()
coils = coils_via_symmetries(curves, currents, nfp, True)

surf = SurfaceRZFourier.from_nphi_ntheta(mpol=5, ntor=5, stellsym=True, nfp=nfp, range="full torus", nphi=64, ntheta=24)
surf.fit_to_curve(ma, 0.7, flip_theta=False)
surfclassifier = SurfaceClassifier(surf, h=0.1, p=2)

bs = BiotSavart(coils)
R0, Z0 = ma.gamma()[0,::2]
ps = SimsoptBfieldProblem(R0=R0, Z0=Z0, Nfp=nfp, mf=bs, interpolate=True, surf=surf)

###############################################################################
# Poincare plot
###############################################################################

proc0_print("Computing the Poincare plot")
phis = [(i)*(2*np.pi/nfp) for i in range(nfp)]

nfieldlines = 30
phis = [0]    #[(i / 4) * (2 * np.pi / nfp) for i in range(4)]
Rs = np.linspace(ma.gamma()[0, 0], ma.gamma()[0, 0] + 0.14, nfieldlines)
Zs = [ma.gamma()[0, 2] for i in range(nfieldlines)]
startconfigs = np.array([[r, z] for r, z in zip(Rs, Zs)])

# Poincare plot
logging.basicConfig()
logger = logging.getLogger('simsopt.field.tracing')
logger.setLevel(1)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
comm_config = np.array_split(startconfigs.reshape(-1,2), comm.Get_size())
comm_config = comm_config[rank]

pplane = poincare(ps._mf_B, comm_config, phis, surfclassifier, tmax = 10000, tol = 1e-13, plot=False, comm=None)

# tys, phi_hits = pplane.tys, pplane.phi_hits

pplane.save("pkl/default.pkl")

###############################################################################
# Saving the plot
###############################################################################

# latexplot_folder = Path("../../latex/images/plots").absolute()
# saving_folder = Path("figs").absolute()

# sys.path.append(str(latexplot_folder))
# from plot_poincare import plot_poincare_simsopt

# fig, ax = plt.subplots()
# plot_poincare_simsopt(pplane.phi_hits, ax)
# fig.savefig