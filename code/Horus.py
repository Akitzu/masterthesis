import numpy as np
from scipy.integrate import solve_ivp
from simsopt.configs import get_ncsx_data
from simsopt.field import (
    MagneticField,
    BiotSavart,
    InterpolatedField,
    coils_via_symmetries,
    SurfaceClassifier,
    compute_fieldlines,
    LevelsetStoppingCriterion,
    plot_poincare_data,
)
from simsopt.util import comm_world
from simsopt.geo import SurfaceRZFourier

from pyoculus.problems import CartesianBfield
from pyoculus.solvers import FixedPoint, PoincarePlot

import matplotlib.pyplot as plt
from matplotlib import cm

### Helper functions for Horus ###


def cyltocart(r, theta, z):
    return np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [-r * np.sin(theta), -r * np.cos(theta), 0],
            [0, 0, 1],
        ]
    )


def carttocyl(x, y, z):
    return np.linalg.inv(cyltocart(x, y, z))


def normalize(v: np.ndarray) -> np.ndarray:
    """Compute the normalized vector of v."""
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


### Simsopt magnetic field problem class ###
class SimsoptBfieldProblem(CartesianBfield):
    def __init__(self, R0, Z0, Nfp, bs):
        super().__init__(R0, Z0, Nfp)

        if not isinstance(bs, MagneticField):
            raise ValueError("bs must be a MagneticField object")

        self.bs = bs

    # The return of the B field for the two following methods is not the same as the calls are :
    #   - CartesianBfield.f_RZ which does :
    #   line 37     B = np.array([self.B(xyz, *args)]).T
    #   - CartesianBfield.f_RZ_tangent which does :
    #   line 68     B, dBdX = self.dBdX(xyz, *args)
    #   line 69     B = np.array(B).T
    # and both should result in a (3,1) array
    def B(self, xyz):
        xyz = np.reshape(xyz, (-1, 3))
        self.bs.set_points(xyz)
        return self.bs.B().flatten()

    def dBdX(self, xyz):
        B = self.B(xyz)
        return [B], self.bs.dB_by_dX().reshape(3, 3)


### Stellerator configurations ###


def ncsx():
    """Get the NCSX stellarator configuration."""
    return stellarator(*get_ncsx_data(), nfp=3)

def stellarator(curves, currents, ma, nfp, **kwargs):
    """Set up a stellarator configuration and returns the magnetic field and the interpolated magnetic field as well as coils and the magnetic axis.

    Args:
        curves (simsopt.CurveXYZFourier): list of curves
        currents (list of simsopt.Current): list of currents
        ma (simsopt.CurveRZFourier): magnetic axis
        nfp (int): number of field periods


        degree (int): degree of the interpolating polynomial
        n (int): number of points in the radial direction
        mpol (int): number of poloidal modes
        ntor (int): number of toroidal modes
        stellsym (bool): whether to exploit stellarator symmetry

    Returns:
        tuple: (Biot-Savart object, InterpolatedField object, (nfp, coils, ma, sc_fieldline))
    """
    options = {"degree": 2, "n": 20, "mpol": 5, "ntor": 5, "stellsym": True}
    options.update(kwargs)

    # Load the NCSX data and create the coils
    coils = coils_via_symmetries(curves, currents, nfp, True)
    curves = [c.curve for c in coils]

    # Create the Biot-Savart object
    bs = BiotSavart(coils)

    # Create the surface and the surface classifier
    s = SurfaceRZFourier.from_nphi_ntheta(
        mpol=options["mpol"],
        ntor=options["ntor"],
        stellsym=options["stellsym"],
        nfp=nfp,
        range="full torus",
        nphi=64,
        ntheta=24,
    )
    s.fit_to_curve(ma, 0.70, flip_theta=False)
    sc_fieldline = SurfaceClassifier(s, h=0.03, p=2)

    # Bounds for the interpolated magnetic field chosen so that the surface is
    # entirely contained in it
    rs = np.linalg.norm(s.gamma()[:, :, 0:2], axis=2)
    zs = s.gamma()[:, :, 2]
    rrange = (np.min(rs), np.max(rs), options["n"])
    phirange = (0, 2 * np.pi / nfp, options["n"] * 2)
    # exploit stellarator symmetry and only consider positive z values:
    zrange = (0, np.max(zs), options["n"] // 2)

    def skip(rs, phis, zs):
        # The RegularGrindInterpolant3D class allows us to specify a function that
        # is used in order to figure out which cells to be skipped.  Internally,
        # the class will evaluate this function on the nodes of the regular mesh,
        # and if *all* of the eight corners are outside the domain, then the cell
        # is skipped.  Since the surface may be curved in a way that for some
        # cells, all mesh nodes are outside the surface, but the surface still
        # intersects with a cell, we need to have a bit of buffer in the signed
        # distance (essentially blowing up the surface a bit), to avoid ignoring
        # cells that shouldn't be ignored
        rphiz = np.asarray([rs, phis, zs]).T.copy()
        dists = sc_fieldline.evaluate_rphiz(rphiz)
        skip = list((dists < -0.05).flatten())
        return skip

    bsh = InterpolatedField(
        bs,
        options["degree"],
        rrange,
        phirange,
        zrange,
        True,
        nfp=nfp,
        stellsym=True,
        skip=skip,
    )

    return bs, bsh, (nfp, coils, ma, sc_fieldline)


### Magnetic field line tracing using solve_ip ###


def trace(bs, tf, xx, **kwargs):
    """Compute the curve of the magnetic field line in 3d in the forward direction from the initial point xx
    using scipy to solve the Initial Value Problem.

    Args:
        bs (simsopt.MagneticField): Magnetic Field object, for instance simsopt.BiotSavart
        tw (tuple): time window (tmax, nsteps)
        xx (np.ndarray): initial point in 3d cartesian coordinates
        t_eval (np.ndarray): time points to evaluate the solution,
            if None, use np.linspace(0, tf, steps)

    Returns:
        np.ndarray: the trace of the magnetic field line
    """
    options = {
        "rtol": 1e-10,
        "atol": 1e-10,
        "t_eval": None,
        "steps": int(tf * 1000),
        "method": "DOP853",
    }
    options.update(kwargs)

    if options["t_eval"] is None:
        options["t_eval"] = np.linspace(0, tf, options["steps"])

    def unit_Bfield(t, xx):
        bs.set_points(xx.reshape((-1, 3)))
        return normalize(bs.B().flatten())

    out = solve_ivp(
        unit_Bfield,
        [0, tf],
        xx,
        t_eval=options["t_eval"],
        method=options["method"],
        rtol=options["rtol"],
        atol=options["atol"],
    )
    return out.y


### Different ways to draw a Poincare plot ###


def poincare(engine, bs, RZstart, phis, sc_fieldline=None, **kwargs):
    if engine == "simsopt":
        return poincare_simsopt(bs, RZstart, phis, sc_fieldline, **kwargs)
    elif engine == "ivp":
        return poincare_ivp(bs, RZstart, phis, **kwargs)


def poincare_simsopt(bs, RZstart, phis, sc_fieldline, **kwargs):
    options = {"tmax": 40000, "tol": 1e-7}
    options.update(kwargs)

    fieldlines_tys, fieldlines_phi_hits = compute_fieldlines(
        bs,
        RZstart[:, 0],
        RZstart[:, 1],
        tmax=options["tmax"],
        tol=options["tol"],
        comm=comm_world,
        phis=phis,
        stopping_criteria=[LevelsetStoppingCriterion(sc_fieldline.dist)],
    )

    if comm_world is None or comm_world.rank == 0:
        fig, axs = plot_poincare_data(fieldlines_phi_hits, phis)

    return fig, axs


def poincare_ivp(bs, RZstart, phis, **kwargs):
    options = {
        "rtol": 1e-10,
        "atol": 1e-10,
        "t_eval": None,
        "steps": int(1000),
        "method": "DOP853",
    }
    options.update(kwargs)

    def Bfield_2D(t, xx):
        xx = xx.reshape((-1, 3))
        bs.set_points(xx)
        B = carttocyl @ bs.B().reshape(3, -1)
        return B[0, 0] / B[1, 0], B[2, 0] / B[1, 0]

    out = solve_ivp(
        Bfield_2D,
        [0, 1],
        RZstart,
        t_eval=phis,
        method=options["method"],
        rtol=options["rtol"],
        atol=options["atol"],
    )


### Convergence domain for the X-O point finders ###


def convergence_domain(ps, Rw, Zw, **kwargs):
    """Compute where the FixedPoint solver converge to in the R-Z plane. Each point from the meshgrid given by the input Rw and Zw is tested for convergence.
    if a point converges, it is assigned a number, otherwise it is assigned -1. The number corresponds to the index of the fixed point in returned list of fixed points.

    Args:
        ps (pyoculus.problems.CartesianBfield): the problem to solve
        Rw (np.ndarray): the R values of the meshgrid
        Zw (np.ndarray): the Z values of the meshgrid

    Keyword Args:
        -- FixedPoint.compute --
        pp (int): the poloidal mode to use
        qq (int): the toroidal mode to use
        sbegin (float): the starting value of the R parameter
        send (float): the ending value of the R parameter
        tol (float): the tolerance of the fixed point finder
        checkonly (bool): whether to use checkonly theta for the Newton RZ\n
        -- Integrator --
        rtol (float): the relative tolerance of the integrator\n
        --- FixedPoint ---
        nrestart (int): the number of restarts for the fixed point finder
        niter (int): the number of iterations for the fixed point finder\n
        -- Comparison --
        eps (float): the tolerance for the comparison with the fixed points

    Returns:
        np.ndarray: the R values of the meshgrid
        np.ndarray: the Z values of the meshgrid
        np.ndarray: the assigned number for each point in the meshgrid
        list: the list of fixed points object (BaseSolver.OutputData)
    """
    options = {
        "pp": 3,
        "qq": 7,
        "sbegin": 1.2,
        "send": 1.9,
        "tol": 1e-4,
        "checkonly": False,
        "eps": 1e-4,
    }
    options.update(kwargs)

    # set up the integrator
    iparams = {"rtol": 1e-7}
    iparams.update(kwargs)

    # set up the point finder
    pparams = {"nrestart": 0, "niter": 30}
    pparams.update(kwargs)

    fp = FixedPoint(ps, pparams, integrator_params=iparams)
    R, Z = np.meshgrid(Rw, Zw)

    assigned_to = list()
    fixed_points = list()

    for r, z in zip(R.flatten(), Z.flatten()):
        fp_result = fp.compute(
            guess=[r, z],
            pp=options["pp"],
            qq=options["qq"],
            sbegin=options["sbegin"],
            send=options["send"],
            tol=options["tol"],
            checkonly=options["checkonly"],
        )

        if fp_result is not None:
            fp_result_xyz = np.array([fp_result.x[0], fp_result.y[0], fp_result.z[0]])
            assigned = False
            for j, fpt in enumerate(fixed_points):
                fpt_xyz = np.array([fpt.x[0], fpt.y[0], fpt.z[0]])
                if np.isclose(fp_result_xyz, fpt_xyz, atol=options['eps']).all():
                    assigned_to.append(j)
                    assigned = True
            if not assigned:
                assigned_to.append(len(fixed_points))
                fixed_points.append(fp_result)
        else:
            assigned_to.append(-1)

    return R, Z, assigned_to, fixed_points


def plot_convergence_domain(R, Z, assigned_to, fixed_points, ax=None, colors=None):
    """Plot the convergence domain for FixedPoint solver in the R-Z plane. If ax is None, a new figure is created,
    otherwise the plot is added to the existing figure.

    Args:
        R (np.ndarray): the R values of the meshgrid
        Z (np.ndarray): the Z values of the meshgrid
        assigned_to (np.ndarray): the assigned number for each point in the meshgrid
        fixed_points (list): the list of fixed points object (BaseSolver.OutputData)\n
        -- Optional --
        ax (matplotlib.axes.Axes): the axes to plot on. Defaults to None.
        colors (np.ndarray): the colors to use. Defaults to COLORS. Should be of dimension (k, 3 or 4) for RGB/RGBA with k at least the number of fixed point plus one.

    Returns:
        tuple: (fig, ax)
    """

    assigned_to = np.array(assigned_to) + 1

    if colors is None:
        colors = cm.rainbow(np.linspace(0, 1, len(fixed_points) + 1))
        colors[:, 3] = 0.8
        colors = np.vstack(([0.5,0.5,0.5,0.5], colors))

    cmap = np.array([colors[j] for j in assigned_to])
    cmap = cmap.reshape(R.shape[0], R.shape[1], cmap.shape[1])

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    ax.pcolormesh(R, Z, cmap, shading="nearest")

    # for r,z in zip(R, Z):
    #     ax.scatter(r, z, color = 'blue', s = 1)

    for i, fpt in enumerate(fixed_points):
        if fpt.GreenesResidue < 0:
            marker = "X"
        elif fpt.GreenesResidue > 0:
            marker = "o"
        else:
            marker = "s"

        ax.scatter(
            fpt.x[0],
            fpt.z[0],
            color=colors[i + 1, :3],
            marker=marker,
            edgecolors="black",
            linewidths=1,
        )

    # # Plot arrows from the meshgrid points to the fixed points they converge to
    # for r, z, a in zip(R.flat, Z.flat, assigned_to.flat):
    #     if a > 0:
    #         fpt = fixed_points[a - 1]
    #         dr = np.array([fpt.x[0] - r, fpt.z[0] - z])
    #         dr = 0.1*dr
    #         ax.arrow(r, z, dr[0], dr[1], color='blue')

    ax.set_aspect("equal")

    return fig, ax