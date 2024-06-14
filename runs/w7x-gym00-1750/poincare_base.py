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
from simsopt.configs import get_w7x_data
from pyoculus.problems import SimsoptBfieldProblem
from pyoculus.solvers import FixedPoint, Manifold
import matplotlib.pyplot as plt
from horus import poincare
import pickle

###############################################################################
# Define the W7X cpnfiguration and set up the pyoculus problem
###############################################################################

nfp = 5  # Number of field periods
curves, currents, ma = get_w7x_data()

# GYM00+1750 currents
currents_gym = [Current(1.109484) * 1e6 for _ in range(5)]
currents_gym.append(Current(-0.3661) * 1e6)
currents_gym.append(Current(-0.3661) * 1e6)

# GYM00+1750 currents but opposite direction
currents_gym_bwd = [Current(-1.109484) * 1e6 for _ in range(5)]
currents_gym_bwd.append(Current(0.3661) * 1e6)
currents_gym_bwd.append(Current(0.3661) * 1e6)

# coils = coils_via_symmetries(curves, currents_gym, 5, True)
# coils = coils_via_symmetries(curves, currents_gym_bwd, 5, True)
coils = coils_via_symmetries(curves, currents, 5, True)

# Surface delimiter
proc0_print("Setting up the surface")
# from pyoculus.problems import surf_from_coils
# surf = surf_from_coils(coils, ncoils=7, mpol=5, ntor=5)

surf = SurfaceRZFourier.from_nphi_ntheta(mpol=5, ntor=5, stellsym=True, nfp=5, range="full torus", nphi=64, ntheta=24)
surf.fit_to_curve(ma, 1.5, flip_theta=False)

surfclassifier = SurfaceClassifier(surf, h=0.1, p=2)
proc0_print("Surface and Classifier set up")

# Setting the problem
proc0_print("Setting up the problem")
R0, _, Z0 = ma.gamma()[0,:]
bs = BiotSavart(coils)
ps = SimsoptBfieldProblem.without_axis([5.98, 0], nfp, bs)
R0, Z0 = ps._R0, ps._Z0
# ps = SimsoptBfieldProblem(R0=R0, Z0=Z0, Nfp=nfp, mf=bs)
ps = SimsoptBfieldProblem(R0=R0, Z0=Z0, Nfp=nfp, mf=bs, interpolate=True, surf=surf)
proc0_print("Problem set up")

###############################################################################
# Poincare plot
###############################################################################

proc0_print("Computing the Poincare plot")
phis = [(i)*(2*np.pi/nfp) for i in range(nfp)]

# #GYM00+1750
# nfieldlines = 20
# p1 = np.array([ps._R0, ps._Z0])
# p2 = np.array([5.69956997, 0.52560335])
# Rs = np.linspace(p1[0], p2[0], nfieldlines)
# Zs = np.linspace(p1[1], p2[1], nfieldlines)
# RZs = np.array([[r, z] for r, z in zip(Rs.flatten(), Zs.flatten())])

# nfieldlines = 30
# p1 = np.array([5.6138, 0.8073])
# p2 = np.array([5.809, 0.675])
# Rs = np.linspace(p1[0], p2[0], nfieldlines)
# Zs = np.linspace(p1[1], p2[1], nfieldlines)
# # Rs, Zs = np.meshgrid(Rs, Zs)
# RZs2 = np.array([[r, z] for r, z in zip(Rs.flatten(), Zs.flatten())])
# startconfigs = np.concatenate((RZs, RZs2))

# Default
nfieldlines = 30
p1 = np.array([ps._R0, ps._Z0])
p2 = np.array([5.553, -1.1])
Rs = np.linspace(p1[0], p2[0], nfieldlines)
Zs = np.linspace(p1[1], p2[1], nfieldlines)
RZs = np.array([[r, z] for r, z in zip(Rs.flatten(), Zs.flatten())])
startconfigs = RZs

# # Poincare plot
logging.basicConfig()
logger = logging.getLogger('simsopt.field.tracing')
logger.setLevel(1)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
comm_config = np.array_split(startconfigs.reshape(-1,2), comm.Get_size())
comm_config = comm_config[rank]

pplane = poincare(ps._mf_B, comm_config, phis, surfclassifier, tmax = 10000, tol = 1e-13, plot=False, comm=None)
# pplane = poincare(ps._mf, comm_config, phis, surfclassifier, tmax = 10000, tol = 1e-13, plot=False, comm=comm_world)

tys, phi_hits = pplane.tys, pplane.phi_hits

# if comm is None or comm.rank == 0:
#     with open('manifold.pkl', 'wb') as f:
#         pickle.dump((tys, phi_hits), f)

# pplane.save("pkl/w7x_gym00_1750.pkl")
pplane.save("pkl/default.pkl")

################################################################################
## Find fixed point, set the manifold and find the clinics
################################################################################

# # set up the integrator
# iparams = dict()
# iparams["rtol"] = 1e-12

# pparams = dict()
# pparams["nrestart"] = 0
# pparams["tol"] = 1e-18
# pparams['niter'] = 100

# fp_x1 = FixedPoint(ps, pparams, integrator_params=iparams)
# fp_x2 = FixedPoint(ps, pparams, integrator_params=iparams)

# fp_x1.compute(guess=[5.69956997, 0.52560335], pp=5, qq=4, sbegin=5.2, send=6.2, checkonly=True)
# fp_x2.compute(guess=[5.883462104879646, 0.6556749703570318], pp=5, qq=4, sbegin=5.2, send=6.2, checkonly=True)

# # Working on manifold
# iparam = dict()
# iparam["rtol"] = 1e-13

# mp = Manifold(ps, fp_x1, fp_x2, integrator_params=iparam)
# mp.choose(signs=[[1, -1], [-1, 1]], order=False)

# mp.onworking = mp.outer
# mp.find_clinic_single(0.0005262477692494645, 0.0011166626256106193, n_s=7, n_u=7)
# mp.find_clinic_single(0.0006491052045732048, 0.0013976539980315229, n_s=7, n_u=6)

###############################################################################
# Manifold plotting
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

# neps = 2*25
# start_eps_s = np.logspace(
#     np.log(eps_s_1) / np.log(lambda_s), np.log(eps_s_3) / np.log(lambda_s),
#     neps, base=lambda_s, endpoint=False)
# start_eps_s = np.concatenate((start_eps_s, [eps_s_2]))
# start_eps_s = np.sort(start_eps_s)

# start_eps_u = np.logspace(
#     np.log(eps_u_1) / np.log(lambda_u), np.log(eps_u_3) / np.log(lambda_u),
#     neps, base=lambda_u, endpoint=False)
# start_eps_u = np.concatenate((start_eps_u, [eps_u_2]))
# start_eps_u = np.sort(start_eps_u)

# rz_start_s = (np.ones((neps+1,1))*rfp_s) + (np.atleast_2d(start_eps_s).T * vector_s)
# rz_start_u = (np.ones((neps+1,1))*rfp_u) + (np.atleast_2d(start_eps_u).T * vector_u)

# # # rz_start_s = np.load("pkl/rz_start_s.npy")
# # # rz_start_u = np.load("pkl/rz_start_u.npy")

# # startconfigs = np.concatenate((rz_start_s, rz_start_u)) 

# proc0_print("Computing the Manifold plot")
# phis = [(i)*(2*np.pi/nfp) for i in range(nfp)]

# # Poincare plot
# logging.basicConfig()
# logger = logging.getLogger('simsopt.field.tracing')
# logger.setLevel(1)

# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
# comm_config = np.array_split(rz_start_s.reshape(-1,2), comm.Get_size())
# # comm_config = np.array_split(rz_start_u.reshape(-1,2), comm.Get_size())
# comm_config = comm_config[rank]

# pplane = poincare(ps._mf_B, comm_config, phis, surfclassifier, tmax = 1000, tol = 1e-13, plot=False, comm=comm)

# if comm.rank == 0:
#     tys, phi_hits = pplane.tys, pplane.phi_hits
#     with open('manifold.pkl', 'wb') as f:
#         pickle.dump((tys, phi_hits), f)

# tys_list = comm.gather(tys, root=0)
# phi_hits_list = comm.gather(phi_hits, root=0)

# if comm_world is None or comm_world.rank == 0:
#     tys = [t for ts in tys_list for t in ts]
#     phi_hits = [phi for phis in phi_hits_list for phi in phis]