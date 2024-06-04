#!/usr/bin/env python

import time
import os
import logging
from pathlib import Path
import numpy as np
import simsoptpp as sopp
from simsopt.geo import CurveHelical, CurveXYZFourier, curves_to_vtk, SurfaceRZFourier
from simsopt.field import Current, Coil, BiotSavart
from simsopt.field import (InterpolatedField, SurfaceClassifier, particles_to_vtk,
                           LevelsetStoppingCriterion,
                           MinRStoppingCriterion, MaxRStoppingCriterion,
                           MinZStoppingCriterion, MaxZStoppingCriterion,
                           )
from simsopt.util import proc0_print, comm_world
from simsopt._core.util import parallel_loop_bounds
import matplotlib.pyplot as plt

####################################################################################################
# Define the LHD-like coils.
# In contrast to LHD, here the coils are infinitesmal filaments rather than
# finite thickness,
# and the helical coils have a "straight" winding law (i.e. no terms other than theta = const1 * phi + const2).
# The coil locations and currents are from the coils file and vmec input files
# sent by Yasuhiro Suzuki in an email on Feb 7, 2002. Also see
# https://ocw.mit.edu/courses/22-68j-superconducting-magnets-spring-2003/0a300450f966033fac5ee5c8107fbe55_lec6rev_3apr03_3.pdf
# https://www-pub.iaea.org/MTCD/publications/PDF/csp_001c/pdf/ov1_4.pdf (table 2)
# Imagawa et al, Fusion Engineering and Design 41 (1998) 253
####################################################################################################

# Coil current values from ~/vmec_equilibria/LHD/LHD/input.r360_g12538_bq+100_ppkf2.b000
# These currents are for the LHD configuration in which the magnetic axis has
# major radius 3.6m. For the straight winding law used here, the major radius
# will generally be different. Note that these currents are per filament in
# Suzuki's coils file, not the total number of Ampere-turns.
extcur = [
    12000.0,
    12000.0,
    12000.0,
    -19600.0,
    -3280.0,
    12250.0,
]

R_major = 3.9  # Major radius of the helical coil
r_minor = 0.975  # Minor radius of the helical coil
turns_per_helical_coil = 300 / 2  # The coils file has 300 filaments for the 2 helical coils combined
current_helical = Current(sum(extcur[:3]) * turns_per_helical_coil)
# In the following lines, 288, 416, and 480 are the number of filaments in Suzuki's coils file:
current_OV = Current(extcur[3] * 288 / 2)
current_IS = Current(extcur[4] * 416 / 2)
current_IV = Current(extcur[5] * 480 / 2)

n0 = 5
l0 = 1
nfp = 10

nquadpoints = 200
order = 1  # Maximum Fourier mode number in the helical coil winding law

curve_helical1 = CurveHelical(
    nquadpoints,
    order,
    n0,
    l0,
    R_major,
    r_minor,
)
curve_helical2 = CurveHelical(
    nquadpoints,
    order,
    n0,
    l0,
    R_major,
    r_minor,
)
# Set helical coil 2 to be shifted in phase by pi relative to helical coil 1:
x = curve_helical2.x
x[0] = np.pi
curve_helical2.x = x

def circular_coil(R0, z):
    nquadpoints = 200
    order = 1
    c = CurveXYZFourier(nquadpoints, order)
    x = c.x
    x[2] = R0
    x[4] = R0
    x[6] = z
    c.x = x
    return c

# LHD coil parameters, from 20240331-02_analyze_LHD_coils_file.py
OV_R = 5.55
OV_z = 1.55

IS_R = 2.82
IS_z = 2.0

IV_R = 1.8
IV_z = 0.8

curve_OVU = circular_coil(OV_R, OV_z)
curve_OVL = circular_coil(OV_R, -OV_z)

curve_ISU = circular_coil(IS_R, IS_z)
curve_ISL = circular_coil(IS_R, -IS_z)

curve_IVU = circular_coil(IV_R, IV_z)
curve_IVL = circular_coil(IV_R, -IV_z)

curves = [
    curve_helical1,
    curve_helical2,
    curve_OVU,
    curve_OVL,
    curve_ISU,
    curve_ISL,
    curve_IVU,
    curve_IVL,
]

curves_to_vtk(curves, __file__ + ".curves", close=True)

coils = [
    Coil(curve_helical1, current_helical),
    Coil(curve_helical2, current_helical),
    Coil(curve_OVU, current_OV),
    Coil(curve_OVL, current_OV),
    Coil(curve_ISU, current_IS),
    Coil(curve_ISL, current_IS),
    Coil(curve_IVU, current_IV),
    Coil(curve_IVL, current_IV),
]
bs = BiotSavart(coils)

bs.set_points(np.array([[R_major, 0, 0]]))
proc0_print("B at (R_major, 0, 0):", bs.B())

####################################################################################################
# End of defining the coils.
# Now set parameters for field line tracing.
####################################################################################################

tmax_fl = 100  # "Time" for field line tracing
tol = 1e-13
degree = 4  # Polynomial degree for interpolating the magnetic field

interpolant_n = 20  # Number of points in each dimension for the interpolant

# Set the range for the interpolant and the stopping conditions for field line tracing:
margin = 1.5
Zmax = r_minor * margin
Rmax = R_major + r_minor * margin
Rmin = R_major - r_minor * margin

# Set initial locations from which field lines will be traced:
nR0 = 40
nphi0 = 39
nfieldlines = nR0 * nphi0
R1D = np.concatenate((np.linspace(2.5, 3.6, nR0), np.linspace(4.15, 5, nR0)))
phi1D = np.linspace(0, 2 * np.pi, nphi0, endpoint=False)

R0, phi0 = np.meshgrid(R1D, phi1D)
X0 = R0 * np.cos(phi0)
Y0 = R0 * np.sin(phi0)
X0_flat = X0.flatten()
Y0_flat = Y0.flatten()

initial_conditions = np.array([X0_flat, Y0_flat, np.zeros_like(X0_flat)]).T

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
classifier_aminor = 1.3
classifier_elongation = 2.5
classifier_elongation_inv = 1 / classifier_elongation
b = classifier_aminor / np.sqrt(classifier_elongation_inv)
surf_classifier.set_rc(0, 0, R_major)
surf_classifier.set_rc(1, 0, 0.5 * b * (classifier_elongation_inv + 1))
surf_classifier.set_zs(1, 0, -0.5 * b * (classifier_elongation_inv + 1))
surf_classifier.set_rc(1, 1, 0.5 * b * (classifier_elongation_inv - 1))
surf_classifier.set_zs(1, 1, 0.5 * b * (classifier_elongation_inv - 1))
surf_classifier.x = surf_classifier.x

surf_classifier.to_vtk(__file__ + 'surf_classifier')
sc_fieldline = SurfaceClassifier(surf_classifier, h=0.2, p=2)

stopping_criteria=[
    MaxZStoppingCriterion(Zmax),
    MinZStoppingCriterion(-Zmax),
    MaxRStoppingCriterion(Rmax),
    MinRStoppingCriterion(Rmin),
    LevelsetStoppingCriterion(sc_fieldline.dist),
]

logging.basicConfig()
logger = logging.getLogger('simsopt.field.tracing')
logger.setLevel(1)

# Same as simsopt's compute_fieldlines, but take the initial locations as
# (x,y,z) values instead of (R, Z) values.
def compute_fieldlines(field, xyz_inits, tmax=200, tol=1e-7, phis=[], stopping_criteria=[], comm=None):
    r"""
    Compute magnetic field lines by solving

    .. math::

        [\dot x, \dot y, \dot z] = B(x, y, z)

    Args:
        field: the magnetic field :math:`B`
        R0: list of radial components of initial points
        Z0: list of vertical components of initial points
        tmax: for how long to trace. will do roughly ``|B|*tmax/(2*pi*r0)`` revolutions of the device
        tol: tolerance for the adaptive ode solver
        phis: list of angles in [0, 2pi] for which intersection with the plane
              corresponding to that phi should be computed
        stopping_criteria: list of stopping criteria, mostly used in
                           combination with the ``LevelsetStoppingCriterion``
                           accessed via :obj:`simsopt.field.tracing.SurfaceClassifier`.

    Returns: 2 element tuple containing
        - ``res_tys``:
            A list of numpy arrays (one for each particle) describing the
            solution over time. The numpy array is of shape (ntimesteps, 4).
            Each row contains the time and
            the position, i.e.`[t, x, y, z]`.
        - ``res_phi_hits``:
            A list of numpy arrays (one for each particle) containing
            information on each time the particle hits one of the phi planes or
            one of the stopping criteria. Each row of the array contains
            `[time, idx, x, y, z]`, where `idx` tells us which of the `phis`
            or `stopping_criteria` was hit.  If `idx>=0`, then `phis[int(idx)]`
            was hit. If `idx<0`, then `stopping_criteria[int(-idx)-1]` was hit.
    """
    nlines = xyz_inits.shape[0]
    # assert len(R0) == len(Z0)
    # assert len(R0) == len(phi0)
    # nlines = len(R0)
    # xyz_inits = np.zeros((nlines, 3))
    # R0 = np.asarray(R0)
    # phi0 = np.asarray(phi0)
    # xyz_inits[:, 0] = R0 * np.cos(phi0)
    # xyz_inits[:, 1] = R0 * np.sin(phi0)
    # xyz_inits[:, 2] = np.asarray(Z0)
    res_tys = []
    res_phi_hits = []
    first, last = parallel_loop_bounds(comm, nlines)
    for i in range(first, last):
        res_ty, res_phi_hit = sopp.fieldline_tracing(
            field, xyz_inits[i, :],
            tmax, tol, phis=phis, stopping_criteria=stopping_criteria)
        res_tys.append(np.asarray(res_ty))
        res_phi_hits.append(np.asarray(res_phi_hit))
        dtavg = res_ty[-1][0]/len(res_ty)
        logger.debug(f"{i+1:3d}/{nlines}, t_final={res_ty[-1][0]}, average timestep {dtavg:.10f}s")
    if comm is not None:
        res_tys = [i for o in comm.allgather(res_tys) for i in o]
        res_phi_hits = [i for o in comm.allgather(res_phi_hits) for i in o]
    return res_tys, res_phi_hits

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
    #R0 = np.ones(nfieldlines) * 1.02
    #Z0 = np.linspace(0, 0.26, nfieldlines)
    #Z0 = np.linspace(0.15, 0.19, nfieldlines)
    #R0 = R02d.flatten()
    #Z0 = Z02d.flatten()
    #phis = [(i/4)*(2*np.pi/nfp) for i in range(4)]
    phis = [(i/4)*(2*np.pi/nfp) for i in range(4 * nfp)]
    fieldlines_tys, fieldlines_phi_hits = compute_fieldlines(
        bfield, initial_conditions, tmax=tmax_fl, tol=tol, comm=comm_world,
        phis=phis, stopping_criteria=stopping_criteria)
    t2 = time.time()
    proc0_print(f"Time for fieldline tracing={t2-t1:.3f}s. Num steps={sum([len(l) for l in fieldlines_tys])//nfieldlines}", flush=True)
    if comm_world is None or comm_world.rank == 0:
        particles_to_vtk(fieldlines_tys, __file__ + f'fieldlines_{label}')
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


# uncomment this to run tracing using the biot savart field (very slow!)
# trace_fieldlines(bs, 'bs')

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


proc0_print('Beginning field line tracing')
# trace_fieldlines(bsh, 'bsh')
#trace_fieldlines(bs, 'bs')

proc0_print("Done. Good bye.")
proc0_print("========================================")

###############################################################################
# Searching the fixed points with pyoculus.
###############################################################################
if comm_world is not None and comm_world.rank != 0:
    comm_world.abort()

proc0_print('Setting up the problem')
from pyoculus.problems import SimsoptBfieldProblem
pyoproblem = SimsoptBfieldProblem.without_axis([3.9, 0], nfp, bs, interpolate=bsh)

# set up the integrator
iparams = dict()
iparams["rtol"] = 1e-10

# set up solver parameters
pparams = dict()
pparams["nrestart"] = 0
pparams["tol"] = 1e-15
# maximum number of newton iterations
pparams['niter'] = 100
pparams['zeta'] = 0.1*np.pi
# pparams['Z'] = 0.

proc0_print('Searching fixed points')
from pyoculus.solvers import FixedPoint
fp_x1 = FixedPoint(pyoproblem, pparams, integrator_params=iparams)
fp_x1.compute(guess=[4.8, 0.0], pp=1, qq=2, sbegin=0.1, send=10, checkonly=True)