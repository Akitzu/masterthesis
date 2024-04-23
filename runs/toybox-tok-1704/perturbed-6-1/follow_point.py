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
    args = parser.parse_args()

    ### Creating the pyoculus problem object
    print("\nCreating the pyoculus problem object\n")

    separatrix = {"type": "circular-current-loop", "amplitude": -10, "R": 6, "Z": -5.5}
    maxwellboltzmann = {"m": 6, "n": -1, "d": np.sqrt(2), "type": "maxwell-boltzmann", "amplitude": 1e-3}

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
    fixedpoint = FixedPoint(pyoproblem, pparams, integrator_params=iparams)

    # find the X-point
    guess = [6.21560891, -4.46981856]
    print(f"Initial guess: {guess}")

    fixedpoint.compute(guess=guess, pp=0, qq=1, sbegin=4, send=9, tol=1e-10)

    if fixedpoint.successful:
        results = [list(p) for p in zip(fixedpoint.x, fixedpoint.y, fixedpoint.z)]
    else:
        raise ValueError("X-point not found")

    fig = pickle.load(open("../path-to-folder/path-to-file.pkl", "rb"))
    ax = fig.gca()
    
    iparams = dict()
    iparams["rtol"] = 1e-12

    manifold = Manifold(fixedpoint, pyoproblem, integrator_params=iparams)
    

    if args.no_save:
        # fig.set_size_inches(10, 6) 
        date = datetime.datetime.now().strftime("%m%d%H%M")
        if args.filename:
            dumpname = args.filename
        else:
            dumpname = f"followpoint_{date}"
        with open(dumpname + ".pkl", "wb") as f:
            pickle.dump(fig, f)

    plt.show()
    
    if args.no_save:
        fig.savefig(dumpname + ".png")