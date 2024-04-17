from pyoculus.problems import AnalyticCylindricalBfield
from pyoculus.solvers import PoincarePlot, FixedPoint, Manifold
import matplotlib.pyplot as plt
import numpy as np
import datetime
import argparse
import pickle


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute the Poincare plot of a perturbed tokamak field")
    parser.add_argument('-ns','--no-save', action='store_false', help='Not saving the plot')
    parser.add_argument('-n', '--filename', type=str, default=None, help='Filename to load the plot')
    parser.add_argument('-p','--compute-poincare', action='store_true', help='Computing the poincare plot')
    parser.add_argument('-a','--amplitude', type=float, default=1e-3, help='Amplitude of the perturbation')
    args = parser.parse_args()

    ### Creating the pyoculus problem object
    print("\nCreating the pyoculus problem object\n")

    separatrix = {"type": "circular-current-loop", "amplitude": -10, "R": 6, "Z": -5.5}
    maxwellboltzmann = {"m": 3, "n": -2, "d": 1.3, "type": "maxwell-boltzmann", "amplitude": args.amplitude}

    # gaussian10 = {"m": 1, "n": 0, "d": 1, "type": "gaussian", "amplitude": 0.01}

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

    ### Finding the X-point
    print("\nFinding the X-point\n")

    # set up the integrator for the FixedPoint
    iparams = dict()
    iparams["rtol"] = 1e-12

    pparams = dict()
    pparams["nrestart"] = 0
    pparams["niter"] = 300

    # set up the FixedPoint object
    fp = FixedPoint(pyoproblem, pparams, integrator_params=iparams)

    # find the X-point
    guess = [6.18, -4.45]
    print(f"Initial guess: {guess}")

    fp.compute(guess=guess, pp=0, qq=1, sbegin=1, send=8, tol=1e-10)

    if fp.successful:
        results = [list(p) for p in zip(fp.x, fp.y, fp.z)]
    else:
        raise ValueError("X-point not found")

    ### Compute the Poincare plot
    if args.compute_poincare:
        print("\nComputing the Poincare plot")

        # set up the integrator for the Poincare
        iparams = dict()
        iparams["rtol"] = 1e-7

        # set up the Poincare plot
        pparams = dict()
        pparams["nPtrj"] = 20
        pparams["nPpts"] = 200
        pparams["zeta"] = 0

        # # Set RZs for the normal (R-only) computation
        # pparams["Rbegin"] = 3.01
        # pparams["Rend"] = 5.5

        # Set RZs for the tweaked (R-Z) computation
        nfieldlines = pparams["nPtrj"] + 1

        # Directly setting the RZs
        # Rs = np.linspace(3.2, 3.15, nfieldlines)
        # Zs = np.linspace(-0.43, -2.5, nfieldlines)
        # RZs = np.array([[r, z] for r, z in zip(Rs, Zs)])

        # Two interval computation opoint to xpoint then xpoint to coilpoint
        n1, n2 = int(np.ceil(nfieldlines / 2)), int(np.floor(nfieldlines / 2))
        xpoint = np.array([results[0][0], results[0][2]])
        opoint = np.array([pyoproblem._R0, pyoproblem._Z0])
        coilpoint = np.array(
            [pyoproblem.perturbations_args[0]["R"], pyoproblem.perturbations_args[0]["Z"]]
        )

        # Simple way from opoint to xpoint then to coilpoint
        # Rs = np.concatenate((np.linspace(opoint[0]+1e4, xpoint[0], n1), np.linspace(xpoint[0], coilpoint[0]-1e-4, n2)))
        # Zs = np.concatenate((np.linspace(opoint[1]+1e4, xpoint[1], n1), np.linspace(xpoint[1], coilpoint[1]-1e-4, n2)))
        # RZs = np.array([[r, z] for r, z in zip(Rs, Zs)])

        # Sophisticated way more around the xpoint
        deps = 0.05
        RZ1 = xpoint + deps * (1 - np.linspace(0, 1, n1)).reshape((n1, 1)) @ (
            opoint - xpoint
        ).reshape((1, 2))
        RZ2 = xpoint + deps * np.linspace(0, 1, n2).reshape((n2, 1)) @ (
            coilpoint - xpoint
        ).reshape((1, 2))
        RZs = np.concatenate((RZ1, RZ2))

        # Set up the Poincare plot object
        pplot = PoincarePlot(pyoproblem, pparams, integrator_params=iparams)

        # # R-only computation
        # pplot.compute()

        # R-Z computation
        pplot.compute(RZs)

        fig, ax = pplot.plot(marker=".", s=1, color="black")
    else:
        fig = pickle.load(open("poincare_041610931.pkl", "rb"))
        ax = fig.gca()
    
    ax.scatter(
            pyoproblem._R0, pyoproblem._Z0, marker="o", edgecolors="black", linewidths=1
        )
    ax.scatter(
        results[0][0], results[0][2], marker="X", edgecolors="black", linewidths=1
    )

    iparams = dict()
    iparams["rtol"] = 1e-12

    manifold = Manifold(fp, pyoproblem, integrator_params=iparams)

    print("\nComputing the manifold\n")
    manifold.compute(nintersect = 10, neps = 300, epsilon=1e-6, directions="u+")
    manifold.compute(nintersect = 5, neps = 300, epsilon=1e-6, directions="u-")
    manifold.compute(nintersect = 10, neps = 100, epsilon=1e-6, directions="s+")
    manifold.compute(nintersect = 5, neps = 100, epsilon=1e-6, directions="s-")

    print("\nPlotting the manifold\n")
    manifold.plot(ax)

    if args.no_save:
        fig.set_size_inches(10, 6)
        date = datetime.datetime.now().strftime("%m%d%H%M")
        if args.filename:
            dumpname = args.filename
        else:
            dumpname = f"manifold_{date}"
        with open(dumpname + ".pkl", "wb") as f:
            pickle.dump(fig, f)

    plt.show()
    
    if args.no_save:
        fig.savefig(dumpname + ".png")