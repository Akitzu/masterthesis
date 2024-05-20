from pyoculus.problems import AnalyticCylindricalBfield
from pyoculus.solvers import PoincarePlot, FixedPoint, Manifold
import matplotlib.pyplot as plt
plt.style.use('lateky')
import numpy as np
import datetime
import pickle
from multiprocessing import Pool


def homoclinics(m, n, amplitude):
    # Set up the problem
    pyoproblem = AnalyticCylindricalBfield(
        6,
        0,
        0.8875,
        0.2
    )

    maxwellboltzmann = {"m": m, "n": n, "d": 1.75/np.sqrt(2), "type": "maxwell-boltzmann", "amplitude": amplitude}
    pyoproblem.add_perturbation(maxwellboltzmann)

    ## Poincare plot
    iparams = dict()
    iparams["rtol"] = 1e-10

    # set up the Poincare plot
    pparams = dict()
    pparams["nPtrj"] = 30
    pparams["nPpts"] = 300
    pparams["zeta"] = 0

    # Set RZs for the normal (R-only) computation
    pparams["Rbegin"] = pyoproblem._R0+1e-3
    pparams["Rend"] = 9

    # Set up the Poincare plot object
    pplot = PoincarePlot(pyoproblem, pparams, integrator_params=iparams)

    # # R-only computation
    pplot.compute()

    fig, ax = pplot.plot(marker=".", s=1, color="black")

    ax.scatter(
            pyoproblem._R0, pyoproblem._Z0, marker="o", edgecolors="black", linewidths=1
        )
    
    ## Fixed point
    iparams = dict()
    iparams["rtol"] = 1e-13

    pparams = dict()
    pparams["nrestart"] = 0
    pparams["niter"] = 300
    pparams['Z'] = 0

    # set up the FixedPoint object
    fixedpoint = FixedPoint(pyoproblem, pparams, integrator_params=iparams)

    # find the X-point
    guess = [4., 0.]
    fixedpoint.compute(guess=guess, pp=2, qq=3, sbegin=2, send=10, tol=1e-10)

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

    manifold.choose(0, 1, sngs=[-1, 1])

    manifold.compute(neps=100, nintersect=7)
    manifold.plot(ax, directions="u+s+")

    ax.set_title(f"amplitude = {maxwellboltzmann['amplitude']}, m = {maxwellboltzmann['m']}, n = {maxwellboltzmann['n']}, d = {maxwellboltzmann['d']:.2f}")

    manifold.find_clinics(n_points=4)
    marker = ["X", "o", "s", "p", "P", "*", "x", "D", "d", "^", "v", "<", ">"]
    for i, clinic in enumerate(manifold.clinics):
        _, eps_u_i = clinic[1:3]

        n_u = 8
        
        # hs_i = manifold.integrate(manifold.rfp_s + eps_s_i * manifold.vector_s, n_s, -1)
        hu_i = manifold.integrate(manifold.rfp_u + eps_u_i * manifold.vector_u, n_u, 1)
        # ax.scatter(hs_i[0,:], hs_i[1,:], marker=marker[i], color="royalblue", edgecolor='cyan', zorder=10)
        ax.scatter(hu_i[0,:], hu_i[1,:], marker=marker[i], color="royalblue", edgecolor='cyan', zorder=10, label=f'$h_{i+1}$')


    fig.set_size_inches(12, 12)
    ax.set_xlabel(r'R [m]')
    ax.set_ylabel(r'Z [m]')
    ax.set_xlim(3, 9)
    ax.set_ylim(-3, 3)

    date = datetime.datetime.now().strftime("%m%d%H%M")
    dumpname = f"area_{m}_{n}_{amplitude:.5f}_{date}"
    with open(dumpname + ".pkl", "wb") as f:
        pickle.dump(fig, f)
    
    fig.savefig(dumpname + ".png")
    fig, ax = None, None
    plt.close()

    areas = manifold.resonance_area()
    return areas

# Define a function to be run in each process
def process_func(args):
    m, n = 3, -2
    amplitude = args
    print(f"Running for m = {m}, n = {n}, amplitude = {amplitude:.5f}")
    try:
        return homoclinics(m, n, amplitude)
    except Exception as e:
        print(e)
        return []


if __name__ == "__main__":
    amp = np.linspace(0., 0.1, 4)
    # Create a pool of 4 processes
    with Pool(4) as p:
        results = p.map(process_func, amp)

    pickle.dump(results, open("areas.pkl", "wb"))