from pyoculus.problems import SimsoptBfieldProblem
from simsopt.field import Current, coils_via_symmetries
from simsopt.configs import get_w7x_data
from pyoculus.solvers import PoincarePlot, FixedPoint, Manifold
from horus import poincare
import matplotlib.pyplot as plt
plt.style.use('lateky')
import numpy as np
import datetime
import pickle
from multiprocessing import Pool
import os
import structlog

log = structlog.get_logger()

def savefig(fig, dumpname):
    os.makedirs("figures", exist_ok=True)

    date = datetime.datetime.now().strftime("%m%d%H%M")
    dumpname = f"{dumpname}_{date}"
    with open("figures/" + dumpname + ".pkl", "wb") as f:
        pickle.dump(fig, f)
    
    fig.savefig("figures/" + dumpname + ".png")

def homoclinics(scaling):
    curves, currents, ma = get_w7x_data()

    default_config = [curr.current for curr in currents]
    log.info(f"Default config: {default_config}")

    # GYM00+1750
    currents = [Current(1.109484 * 1e6) for _ in range(5)]
    currents.append(Current(-0.3661 * 1e6))
    currents.append(Current(-0.3661 * 1e6))
    gym_config = [curr.current for curr in currents]
    log.info(f"Gym config: {gym_config}")

    # Create the intermediate currents
    currents = [Current((1-scaling)*currD + scaling*currG) for currD, currG in zip(default_config, gym_config)]
    log.info(f"Actual config: {[curr.current for curr in currents]}")
    
    # Initialize the problem
    coils = coils_via_symmetries(curves, currents, 5, True)
    R0, _, Z0 = ma.gamma()[0,:]
    pyoproblem = SimsoptBfieldProblem.from_coils(R0=R0, Z0=Z0, Nfp=5, coils=coils, interpolate=True, ncoils=7, mpol=7, ntor=7, n=40)

    log.info(pyoproblem._mf_B.estimate_error_B(10000))

    ## Poincare plot
    phis = [0]

    nfieldlines = 10
    Rs = np.linspace(6.05, 6.2, nfieldlines)
    Zs = [ma.gamma()[0, 2] for _ in range(nfieldlines)]
    RZs = np.array([[r, z] for r, z in zip(Rs, Zs)])

    nfieldlines = 10
    p1 = np.array([5.6144507858315915, -0.8067790944375764])
    p2 = np.array([5.78, -0.6])
    Rs = np.linspace(p1[0], p2[0], nfieldlines)
    Zs = np.linspace(p1[1], p2[1], nfieldlines)
    Rs, Zs = np.meshgrid(Rs, Zs)
    RZs2 = np.array([[r, z] for r, z in zip(Rs.flatten(), Zs.flatten())])

    RZs = np.concatenate((RZs, RZs2))

    pplane = poincare(pyoproblem._mf_B, RZs, phis, pyoproblem.surfclassifier, tol = 1e-9, plot = False)
    fig, ax = pplane.plot(phis, xlims = [5.2, 6.4], ylims = [-1.3, 1.3])

    # Save the plot
    dumpname = f"poincare_{scaling:.5f}"
    savefig(fig, dumpname)

    ### Finding the X-point
    log.info("\nFinding the X-point\n")

    # set up the integrator for the FixedPoint
    iparams = dict()
    iparams["rtol"] = 1e-9

    pparams = dict()
    pparams["nrestart"] = 0
    pparams["tol"] = 1e-13
    pparams['niter'] = 100

    # set up the FixedPoint object
    fixedpoint = FixedPoint(pyoproblem, pparams, integrator_params=iparams)

    # find the X-point
    guess = [5.7, 0.5]
    log.info(f"Initial guess: {guess}")

    fixedpoint.compute(guess=guess, pp=5, qq=4, sbegin=5.2, send=6.2, checkonly=True)

    if fixedpoint.successful:
        results = [list(p) for p in zip(fixedpoint.x, fixedpoint.y, fixedpoint.z)]
    else:
        raise ValueError("X-point not found")

    for rr in results:
        ax.scatter(rr[0], rr[2], marker="X", edgecolors="black", linewidths=1)

    # Manifold
    iparams = dict()
    iparams["rtol"] = 1e-13
    manifold = Manifold(fixedpoint, pyoproblem, integrator_params=iparams)

    manifold.choose(0, 1, ["u+","s+"], [1, 1])

    ax.set_title(f"Lambda : {scaling}")

    log.info("\nFinding the homoclinic points\n")
    manifold.find_clinics(n_points=2)
    
    n_s = manifold.find_clinic_configuration['n_s'] - 1
    n_u = manifold.find_clinic_configuration['n_u']

    log.info(f"\nComputing the manifold\n")
    manifold.compute(neps=300, nintersect= n_s + n_u)
    manifold.plot(ax, directions="u+s+")
    
    marker = ["X", "o", "s", "p", "P", "*", "x", "D", "d", "^", "v", "<", ">"]
    for i, clinic in enumerate(manifold.clinics):
        _, eps_u_i = clinic[1:3]
        
        # hs_i = manifold.integrate(manifold.rfp_s + eps_s_i * manifold.vector_s, n_s, -1)
        hu_i = manifold.integrate(manifold.rfp_u + eps_u_i * manifold.vector_u, n_u + n_s, 1)
        # ax.scatter(hs_i[0,:], hs_i[1,:], marker=marker[i], color="royalblue", edgecolor='cyan', zorder=10)
        ax.scatter(hu_i[0,:], hu_i[1,:], marker=marker[i], color="royalblue", edgecolor='cyan', zorder=10, label=f'$h_{i+1}$')

    fig.set_size_inches(12, 12)
    ax.set_xlabel(r'R [m]')
    ax.set_ylabel(r'Z [m]')

    dumpname = f"clinics_{scaling:.5f}"
    savefig(fig, dumpname)

    fig, ax = None, None
    plt.close()

    areas = manifold.resonance_area()
    return areas

# Define a function to be run in each process
def process_func(args):
    log.info(f"Running for scaling = {args:.5f}")
    try:
        return homoclinics(args)
    except Exception as e:
        log.error(e)
        return []


if __name__ == "__main__":
    amp = np.linspace(0, 1, 3)
    # Create a pool of 4 processes
    with Pool(4) as p:
        results = p.map(process_func, amp)

    pickle.dump(results, open("areas.pkl", "wb"))