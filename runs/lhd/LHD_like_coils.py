#!/usr/bin/env python

import time
import os
import logging
from pathlib import Path
import numpy as np
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
from pyoculus.problems import SimsoptBfieldProblem
from pyoculus.solvers import FixedPoint
import matplotlib.pyplot as plt

###############################################################################
# Define the LHD-like coils.
###############################################################################

coils = load_coils_from_makegrid_file('LHD_data/lhd.coils.txt', order=20)
bs = BiotSavart(coils)
R_major = 3.63  # Major radius of the helical coil
r_minor = 0.975  # Minor radius of the helical coil
nfp = 10  # Number of field periods

###############################################################################
# Set up the stopping criteria and initial conditions for field line tracing.
# Thanks to Matt Landreman for providing this code.
###############################################################################

tmax_fl = 100  # "Time" for field line tracing
tol = 1e-13
degree = 4  # Polynomial degree for interpolating the magnetic field

interpolant_n = 20  # Number of points in each dimension for the interpolant

# Set the range for the interpolant and the stopping conditions for field line tracing:
margin = 2
Zmax = r_minor * margin
Rmax = R_major + r_minor * margin
Rmin = R_major - r_minor * margin

# Set initial locations from which field lines will be traced:
nfieldlines = 60
p1 = np.array([3.63, 0.])
p2 = np.array([3.64, 1.3])
Rs = np.linspace(p1[0], p2[0], nfieldlines)
Zs = np.linspace(p1[1], p2[1], nfieldlines)
initial_conditions = np.array([[r, z] for r, z in zip(Rs.flatten(), Zs.flatten())])

bottom_str = os.path.abspath(__file__) + f"  tol:{tol}  interpolant_n:{interpolant_n}  tmax:{tmax_fl}  nfieldlines: {nfieldlines} degree:{degree}"

# Create a rotating-ellipse surface to use as a stopping condition for field line tracing.
surf_classifier = SurfaceRZFourier.from_nphi_ntheta(
    mpol=1,
    ntor=1,
    nfp=nfp,
    nphi=400,
    ntheta=60,
    range="full torus",
)

classifier_aminor = 2
classifier_elongation = 1.5

classifier_elongation_inv = 1 / classifier_elongation
b = classifier_aminor / np.sqrt(classifier_elongation_inv)
surf_classifier.set_rc(0, 0, R_major)
surf_classifier.set_rc(1, 0, 0.5 * b * (classifier_elongation_inv + 1))
surf_classifier.set_zs(1, 0, -0.5 * b * (classifier_elongation_inv + 1))
surf_classifier.set_rc(1, 1, 0.5 * b * (classifier_elongation_inv - 1))
surf_classifier.set_zs(1, 1, 0.5 * b * (classifier_elongation_inv - 1))
surf_classifier.x = surf_classifier.x

#surf_classifier.to_vtk(__file__ + 'surf_classifier')
sc_fieldline = SurfaceClassifier(surf_classifier, h=0.2, p=2)

stopping_criteria=[
    MaxZStoppingCriterion(Zmax),
    MinZStoppingCriterion(-Zmax),
    MaxRStoppingCriterion(Rmax),
    MinRStoppingCriterion(Rmin),
    LevelsetStoppingCriterion(sc_fieldline.dist),
]

# Set domain for the interpolant:
rrange = (Rmin, Rmax, interpolant_n)
phirange = (0, 2 * np.pi / nfp, interpolant_n * 2)
# exploit stellarator symmetry and only consider positive z values:
zrange = (0, Zmax, interpolant_n//2)

proc0_print('Initializing InterpolatedField')
bsh = InterpolatedField(
    bs, degree, rrange, phirange, zrange, True, nfp=nfp, stellsym=True, #skip=skip
)
proc0_print('Done initializing InterpolatedField.')

logging.basicConfig()
logger = logging.getLogger('simsopt.field.tracing')
logger.setLevel(1)

# Same as simsopt's plot_poincare_data, but with consistent colors between
# subplots, also plotting stellarator-symmetric points, and exploiting nfp
# symmetry to plot more points for the same length of field line tracing.
def plot_poincare_data(fieldlines_phi_hits, phis, filename, mark_lost=False, aspect='equal', dpi=300, xlims=None, 
                       ylims=None, s=2, marker='o'):
    """
    Create a poincare plot. Usage:

    .. code-block::

        phis = np.linspace(0, 2*np.pi/nfp, nphis, endpoint=False)
        res_tys, res_phi_hits = compute_fieldlines(
            bsh, R0, Z0, tmax=1000, phis=phis, stopping_criteria=[])
        plot_poincare_data(res_phi_hits, phis, '/tmp/fieldlines.png')

    Requires matplotlib to be installed.

    """
    import matplotlib.pyplot as plt
    from math import ceil, sqrt
    plt.rc('font', size=7)
    #nrowcol = ceil(sqrt(len(phis)))
    nrowcol = 2
    plt.figure()
    fig, axs = plt.subplots(nrowcol, nrowcol, figsize=(8, 5))
    for ax in axs.ravel():
        ax.set_aspect(aspect)
    color = None
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    for i in range(4):
        row = i//nrowcol
        col = i % nrowcol

        # if passed a surface, plot the plasma surface outline
        for surf in [surf_classifier]:
            cross_section = surf.cross_section(phi=phis[i])
            r_interp = np.sqrt(cross_section[:, 0] ** 2 + cross_section[:, 1] ** 2)
            z_interp = cross_section[:, 2]
            g = 0.0
            axs[row, col].plot(r_interp, z_interp, linewidth=1, c=(g, g, g))

        axs[row, col].grid(True, linewidth=0.5)
            
        if i != 4 - 1:
            axs[row, col].set_title(f"$\\phi = {phis[i]/np.pi:.3f}\\pi$ ", loc='left', y=0.0)
        else:
            axs[row, col].set_title(f"$\\phi = {phis[i]/np.pi:.3f}\\pi$ ", loc='right', y=0.0)
        if row == nrowcol - 1:
            axs[row, col].set_xlabel("$R$")
        if col == 0:
            axs[row, col].set_ylabel("$Z$")
        if col == 1:
            axs[row, col].set_yticklabels([])
        if xlims is not None:
            axs[row, col].set_xlim(xlims)
        if ylims is not None:
            axs[row, col].set_ylim(ylims)
        for j in range(len(fieldlines_phi_hits)):
            if fieldlines_phi_hits[j].size == 0:
                continue
            if mark_lost:
                lost = fieldlines_phi_hits[j][-1, 1] < 0
                color = 'r' if lost else 'g'
            phi_hit_codes = fieldlines_phi_hits[j][:, 1]
            condition = np.logical_and(phi_hit_codes >= 0, np.mod(phi_hit_codes, 4) == i)
            indices = np.where(condition)[0]
            data_this_phi = fieldlines_phi_hits[j][indices, :]
            if data_this_phi.size == 0:
                continue
            r = np.sqrt(data_this_phi[:, 2]**2+data_this_phi[:, 3]**2)
            color = colors[j % len(colors)]
            axs[row, col].scatter(r, data_this_phi[:, 4], marker=marker, s=s, linewidths=0, c=color)

            # stellarator-symmetric points:
            new_row = row
            if i == 1 or i == 3:
                new_row = 1 - row

            axs[new_row, col].scatter(r, -data_this_phi[:, 4], marker=marker, s=s, linewidths=0, c=color)


        #plt.rc('axes', axisbelow=True)

    #plt.figtext(0.5, 0.995, os.path.abspath(coils_filename), ha="center", va="top", fontsize=4)
    plt.figtext(0.5, 0.005, bottom_str, ha="center", va="bottom", fontsize=6)
    plt.tight_layout()
    plt.savefig(filename, dpi=dpi)

def trace_fieldlines(bfield, label):
    t1 = time.time()
    phis = [(i/4)*(2*np.pi/nfp) for i in range(4 * nfp)]
    fieldlines_tys, fieldlines_phi_hits = compute_fieldlines(
        bfield, initial_conditions[:,0], initial_conditions[:,1], tmax=tmax_fl, tol=tol, comm=comm_world,
        phis=phis, stopping_criteria=stopping_criteria)
    t2 = time.time()
    proc0_print(f"Time for fieldline tracing={t2-t1:.3f}s. Num steps={sum([len(l) for l in fieldlines_tys])//nfieldlines}", flush=True)
    if comm_world is None or comm_world.rank == 0:
        # particles_to_vtk(fieldlines_tys, __file__ + f'fieldlines_{label}')
        radius = margin
        plot_poincare_data(
            fieldlines_phi_hits, 
            phis, __file__ + f'_{label}.png', 
            dpi=300, 
            xlims=[R_major - radius, R_major + radius], 
            ylims=[-radius, radius],
            s=1,
            #surf=surf1_Poincare,
        )


import datetime
label = "bsh_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# trace_fieldlines(bsh, label)

###############################################################################
# Searching the fixed points with pyoculus.
###############################################################################
if comm_world is not None and comm_world.rank != 0:
    comm_world.abort()

proc0_print('Setting up the problem')
pyoproblem = SimsoptBfieldProblem.without_axis([3.6, 0], nfp, bs)

# set up the integrator
iparams = dict()
iparams["type"] = "dop853"
iparams["rtol"] = 1e-10
# iparams["min_step"] = 1e-16
# iparams["verbosity "] = 1

# set up solver parameters
pparams = dict()
pparams["nrestart"] = 0
pparams["tol"] = 1e-15
# maximum number of newton iterations
pparams['niter'] = 100
pparams['zeta'] = 0.
# pparams['Z'] = 0.

proc0_print('Searching fixed points')
fp_x1 = FixedPoint(pyoproblem, pparams, integrator_params=iparams)
# fp_x1.compute(guess=[, 0.7], pp=1, qq=2, sbegin=0.1, send=10, checkonly=True)
# fp_x1.compute(guess=[, 0.7], pp=1, qq=2, sbegin=0.1, send=10, checkonly=True)

from pyoculus.integrators import RKIntegrator
iparams = dict()
iparams["rtol"] = 1e-13
iparams["ode"] = pyoproblem.f_RZ
iparams["type"] = "dop853"

integrator = RKIntegrator(iparams)

def evolution(rz, phi0 = 0, dphi = 2*2*np.pi/10):
    integrator.set_initial_value(phi0, rz)
    rz_e = integrator.integrate(phi0+dphi)
    return rz_e # - rz

def Bphi(r, z, phi = 0):
    xyz = np.array([r*np.cos(phi), r*np.sin(phi), z])
    B = pyoproblem.B(xyz)
    return np.matmul(pyoproblem._inv_Jacobian(r, phi, z), B)[1]

# from scipy.optimize import root
# root(evolution, [3.6, 1])