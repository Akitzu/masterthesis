from pyoculus.problems import AnalyticCylindricalBfield
from pyoculus.solvers import PoincarePlot, FixedPoint, Manifold
import matplotlib.pyplot as plt
import numpy as np
import datetime
import pickle
from multiprocessing import Pool


def homoclinics(m, n, amplitude):
    ### Creating the pyoculus problem object
    print("\nCreating the pyoculus problem object\n")

    separatrix = {"type": "circular-current-loop", "amplitude": -10, "R": 6, "Z": -5.5}
    maxwellboltzmann = {"m": m, "n": n, "d": np.sqrt(2), "type": "maxwell-boltzmann", "amplitude": amplitude}
    gaussian = {"m": m, "n": n, "sigma": np.sqrt(2), "mu": 2., "type": "maxwell-boltzmann", "amplitude": amplitude}

    # Creating the pyoculus problem object, adding the perturbation here use the R, Z provided as center point
    pyoproblem = AnalyticCylindricalBfield.without_axis(
        6,
        0,
        0.91,
        0.6,
        perturbations_args=[separatrix],
        Rbegin=1,
        Rend=8,
        niter=800,
        guess=[6.41, -0.7],
        tol=1e-9,
    )

    # # Adding perturbation after the object is created uses the found axis as center point
    pyoproblem.add_perturbation(maxwellboltzmann)

    # breakpoint()

    ### Finding the X-point
    print("\nFinding the X-point\n")

    # set up the integrator for the FixedPoint
    iparams = dict()
    iparams["rtol"] = 1e-12

    pparams = dict()
    pparams["nrestart"] = 0
    pparams["niter"] = 300

    # set up the FixedPoint object
    fixedpoint = FixedPoint(pyoproblem, pparams, integrator_params=iparams)

    # find the X-point
    guess = [6.21560891, -4.46981856]
    print(f"Initial guess: {guess}")

    fixedpoint.compute(guess=guess, pp=0, qq=1, sbegin=4, send=9, tol=1e-10)

    if fixedpoint.successful:
        results = [list(p) for p in zip(fixedpoint.x, fixedpoint.y, fixedpoint.z)]
    else:
        raise ValueError("X-point not found")

    ### Compute the Poincare plot
    print("\nComputing the Poincare plot")

    # set up the integrator for the Poincare
    iparams = dict()
    iparams["rtol"] = 1e-10

    # set up the Poincare plot
    pparams = dict()
    pparams["nPtrj"] = 20
    pparams["nPpts"] = 200
    pparams["zeta"] = 0

    # # Set RZs for the normal (R-only) computation
    # pparams["Rbegin"] = 3.01
    # pparams["Rend"] = 5.5

    # Directly setting the RZs
    # nfieldlines = pparams["nPtrj"] + 1
    # Rs = np.linspace(3.2, 3.15, nfieldlines)
    # Zs = np.linspace(-0.43, -2.5, nfieldlines)
    # RZs = np.array([[r, z] for r, z in zip(Rs, Zs)])

    # Set RZs for the tweaked (R-Z) computation
    frac_nf_1 = 3/4
    nfieldlines_1, nfieldlines_2 = int(np.ceil(frac_nf_1*pparams["nPtrj"])), int(np.floor((1-frac_nf_1)*pparams["nPtrj"]))+1

    # Two interval computation opoint to xpoint then xpoint to coilpoint
    xpoint = np.array([results[0][0], results[0][2]])
    opoint = np.array([pyoproblem._R0, pyoproblem._Z0])
    coilpoint = np.array(
        [pyoproblem.perturbations_args[0]["R"], pyoproblem.perturbations_args[0]["Z"]]
    )

    # Sophisticated way more around the xpoint
    deps = 0.05
    frac_n1 = 1/2
    n1, n2 = int(np.ceil(frac_n1 * nfieldlines_2)), int(np.floor((1 - frac_n1) * nfieldlines_2))
    RZ1 = xpoint + deps * (1 - np.linspace(0, 1, n1)).reshape((n1, 1)) @ (
        opoint - xpoint
    ).reshape((1, 2))
    RZ2 = xpoint + deps * np.linspace(0, 1, n2).reshape((n2, 1)) @ (
        coilpoint - xpoint
    ).reshape((1, 2))
    RZs_2 = np.concatenate((RZ1, RZ2))

    # Simple way from opoint to xpoint then to coilpoint
    frac_n1 = 3/4
    n1, n2 = int(np.ceil(frac_n1 * nfieldlines_1)), int(np.floor((1 - frac_n1) * nfieldlines_1))
    Rs = np.concatenate((np.linspace(opoint[0]+1e-4, xpoint[0]-deps, n1), np.linspace(xpoint[0]+deps, coilpoint[0]-1e-4, n2)))
    Zs = np.concatenate((np.linspace(opoint[1]+1e-4, xpoint[1]-deps, n1), np.linspace(xpoint[1]+deps, coilpoint[1]-1e-4, n2)))
    RZs_1 = np.array([[r, z] for r, z in zip(Rs, Zs)])
    
    # Combine the two sets of RZs
    RZs = np.concatenate((RZs_1, RZs_2))

    # Set up the Poincare plot object
    pplot = PoincarePlot(pyoproblem, pparams, integrator_params=iparams)

    # # R-only computation
    # pplot.compute()

    # R-Z computation
    pplot.compute(RZs)

    fig, ax = pplot.plot(marker=".", s=1, color="black")
    
    ax.scatter(
            pyoproblem._R0, pyoproblem._Z0, marker="o", edgecolors="black", linewidths=1
        )
    ax.scatter(
        results[0][0], results[0][2], marker="X", edgecolors="black", linewidths=1
    )

    iparams = dict()
    iparams["rtol"] = 1e-12

    manifold = Manifold(fixedpoint, pyoproblem, integrator_params=iparams)
    
    # Choose the tangles to work with
    manifold.choose()

    # Find the homoclinic points
    eps_s_1, eps_u_1 = manifold.find_homoclinic(1e-6, 1e-6, n_s = 7, n_u = 6)
    hs_1 = manifold.integrate(manifold.rfp_s + eps_s_1 * manifold.vector_s, 7, -1)
    hu_1 = manifold.integrate(manifold.rfp_u + eps_u_1 * manifold.vector_u, 6, 1)
    ax.scatter(hs_1[0,:], hs_1[1,:], marker="x", color="purple", zorder=10)
    ax.scatter(hu_1[0,:], hu_1[1,:], marker="x", color="blue", zorder=10)


    marker = ["+", "o", "s", "p", "P", "*", "X", "D", "d", "^", "v", "<", ">", "1", "2", "3", "4", "8", "h", "H", "D", "d", "|", "_"]
    
    for i in range(1, 2*np.abs(n)):
        guess_i = [eps_s_1*np.power(manifold.lambda_s, i/(2*np.abs(n))), eps_u_1*np.power(manifold.lambda_u, i/(2*np.abs(n)))]
        print(f"{i}th initial guess: {guess_i}")
        eps_s_n, eps_u_n = manifold.find_homoclinic(guess_i[0], guess_i[1], n_s = 7, n_u = 6)

        hs_i = manifold.integrate(manifold.rfp_s + eps_s_n * manifold.vector_s, 7, -1)
        hu_i = manifold.integrate(manifold.rfp_u + eps_u_n * manifold.vector_u, 6, 1)
        
        # Plot the homoclinic points
        print("\nPlotting homoclinic points")

        ax.scatter(hs_i[0,:], hs_i[1,:], marker=marker[i], color="purple", zorder=10)
        ax.scatter(hu_i[0,:], hu_i[1,:], marker=marker[i], color="blue", zorder=10)

    print("\nComputing the manifold\n")
    manifold.compute(nintersect = 9, neps = 300, epsilon=1e-7)

    print("\nPlotting the manifold\n")
    manifold.plot(ax, directions="u+s+")
    ax.set_title(f"amplitude = {maxwellboltzmann['amplitude']}, m = {maxwellboltzmann['m']}, n = {maxwellboltzmann['n']}, d = {maxwellboltzmann['d']:.2f}")
    
    fig.set_size_inches(10, 10)
    date = datetime.datetime.now().strftime("%m%d%H%M")
    dumpname = f"homoclinics_{m}_{n}_{amplitude:.5f}_{date}"
    with open(dumpname + ".pkl", "wb") as f:
        pickle.dump(fig, f)
    
    fig.savefig(dumpname + ".png")
    fig, ax = None, None
    plt.close()

# Define a function to be run in each process
def process_func(args):
    m, n = args
    amplitude = 0.02
    print(f"Running for m = {m}, n = {n}, amplitude = {amplitude:.5f}")
    try:
        homoclinics(m, n, amplitude)
    except Exception as e:
        print(e)


if __name__ == "__main__":
    # ms = np.arange(1, 20)
    # ns = -1*np.ones_like(ms)
    # ms = [2,2,2,2,5,5,5,5,10,10,10,10,13,13,13,13,17,17,17,17]
    # ns = [-1,-2,-3,-4,-1,-2,-3,-4,-1,-2,-3,-4,-1,-2,-3,-4,-1,-2,-3,-4]
    
    ns = -np.arange(1, 20)
    ms = 2*np.ones_like(ns)

    # Create a pool of 4 processes
    with Pool(4) as p:
        p.map(process_func, zip(ms, ns))