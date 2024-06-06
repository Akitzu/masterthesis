#!/usr/bin/env python

import time
import os
import logging
from pathlib import Path
import numpy as np
from mpi4py import MPI
import simsoptpp as sopp
from simsopt.geo import SurfaceRZFourier # CurveHelical, CurveXYZFourier, curves_to_vtk
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

###############################################################################
# Define the W7X cpnfiguration.
###############################################################################

nfp = 5  # Number of field periods
curves, currents, ma = get_w7x_data()

# GYM00+1750 currents
currents = [Current(1.109484) * 1e6 for _ in range(5)]
currents.append(Current(-0.3661) * 1e6)
currents.append(Current(-0.3661) * 1e6)

coils = coils_via_symmetries(curves, currents, 5, True)

R0, _, Z0 = ma.gamma()[0,:]
bs = BiotSavart(coils)
ps = SimsoptBfieldProblem.without_axis([5.98, 0], nfp, bs)
R0, Z0 = ps._R0, ps._Z0
ps = SimsoptBfieldProblem(R0=R0, Z0=Z0, Nfp=nfp, mf=bs)

################################################################################
## Fixed point and Manifold
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

fp_x1.compute(guess=[5.6995, 0.5256], pp=5, qq=4, sbegin=5.2, send=6.2, checkonly=True)
fp_x2.compute(guess=[5.883462104879646, 0.6556749703570318], pp=5, qq=4, sbegin=5.2, send=6.2, checkonly=True)

# Working on manifold
iparam = dict()
iparam["rtol"] = 1e-13

mp = Manifold(ps, fp_x1, fp_x2, integrator_params=iparam)
mp.choose(signs=[[1, -1], [-1, 1]], order=False)

neps = 200
startconfigs = np.empty((4, neps, 2))
for ii, onworking in enumerate([mp.inner, mp.outer]):
    # eps_s = mp.find_epsilon(onworking["rfp_s"], onworking["vector_s"], 1e-2, -1)
    # eps_u = mp.find_epsilon(onworking["rfp_u"], onworking["vector_u"], 1e-2)
    startconfigs[2*ii,:] = mp.start_config(1e-4, onworking["rfp_s"], onworking["lambda_s"], onworking["vector_s"], neps, -1)
    startconfigs[2*ii+1,:] = mp.start_config(1e-4, onworking["rfp_u"], onworking["lambda_u"], onworking["vector_u"], neps)

###############################################################################
# Manifold plotting
###############################################################################

proc0_print("Computing the Manifold plot")
phis = [(i)*(2*np.pi/nfp) for i in range(nfp)]

# Poincare plot
logging.basicConfig()
logger = logging.getLogger('simsopt.field.tracing')
logger.setLevel(1)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
comm_config = np.array_split(startconfigs.reshape(-1,2), comm.Get_size())
comm_config = comm_config[rank]

#from pyoculus.problems import surf_from_coils
from simsopt.geo import SurfaceClassifier
#surf = surf_from_coils(coils, ncoils=7, mpol=5, ntor=5)

surf = SurfaceRZFourier.from_nphi_ntheta(mpol=5, ntor=5, stellsym=True, nfp=5, range="full torus", nphi=64, ntheta=24)
surf.fit_to_curve(ma, 1.5, flip_theta=False)

surfclassifier = SurfaceClassifier(surf, h=0.03, p=2) 

pplane = poincare(ps._mf_B, comm_config, phis, surfclassifier, tmax = 10000, tol = 1e-13, plot=False, comm=comm_world)
tys, phi_hits = pplane.tys, pplane.phi_hits

tys_list = comm.gather(tys, root=0)
phi_hits_list = comm.gather(phi_hits, root=0)


if comm_world is None or comm_world.rank == 0:
    tys = [t for ts in tys_list for t in ts]
    phi_hits = [phi for phis in phi_hits_list for phi in phis]
    with open('manifold.pkl', 'wb') as f:
        pickle.dump((tys, phi_hits), f)
    
# fig, ax = pplane.plot([0])
# ax = ax[0,0]
# plt.show()

# ###############################################################################
# # Poincare plot
# ###############################################################################

# proc0_print("Computing the Poincare plot")
# phis = [(i)*(2*np.pi/nfp) for i in range(nfp)]

# nfieldlines = 60
# p1 = np.array([ps._R0, ps._Z0])
# p2 = np.array([5.73, -0.669])
# Rs = np.linspace(p1[0], p2[0], nfieldlines)
# Zs = np.linspace(p1[1], p2[1], nfieldlines)
# RZs = np.array([[r, z] for r, z in zip(Rs.flatten(), Zs.flatten())])

# nfieldlines = 10
# p1 = np.array([5.6144507858315915, -0.8067790944375764])
# p2 = np.array([5.78, -0.6])
# Rs = np.linspace(p1[0], p2[0], nfieldlines)
# Zs = np.linspace(p1[1], p2[1], nfieldlines)
# Rs, Zs = np.meshgrid(Rs, Zs)
# RZs2 = np.array([[r, z] for r, z in zip(Rs.flatten(), Zs.flatten())])

# RZs = np.concatenate((RZs, RZs2))

# # Poincare plot
# logging.basicConfig()
# logger = logging.getLogger('simsopt.field.tracing')
# logger.setLevel(1)

# pplane = poincare(ps._mf_B, RZs, phis, ps.surfclassifier, tmax = 10000, tol = 1e-13, plot=False, comm=comm_world)
# fig, ax = pplane.plot([0])
# ax = ax[0,0]

# pplane.save("poincare.pkl")
# plt.show()
