from pyoculus.problems import AnalyticCylindricalBfield
from pyoculus.solvers import PoincarePlot, FixedPoint
import numpy as np
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute the Poincare plot of a perturbed tokamak field")
    parser.add_argument('--save', type=bool, default=True, help='Saving the plot')
    args = parser.parse_args()

    ### Creating the pyoculus problem object
    print("\nCreating the pyoculus problem object\n")

    separatrix = {"type": "circular-current-loop", "amplitude": -10, "R": 6, "Z": -5.5}
    
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
    # pyoproblem = AnalyticCylindricalBfield(6, 0, 0.91, 0.6, perturbations_args=[separatrix])

    ### Compute the Poincare plot
    print("\nComputing the Poincare plot\n")

    # set up the integrator for the Poincare
    iparams = dict()
    iparams["rtol"] = 1e-10

    # set up the Poincare plot
    pparams = dict()
    pparams["nPtrj"] = 50
    pparams["nPpts"] = 200
    pparams["zeta"] = 0

    # Set RZs for the normal (R-only) computation
    pparams["Z"] = pyoproblem._Z0
    pparams["Rbegin"] = pyoproblem._R0+1e-3
    pparams["Rend"] = 9.2

    pplot = PoincarePlot(pyoproblem, pparams, integrator_params=iparams)
    pplot.compute()

    # Iota plot
    rs = np.linspace(pparams["Rbegin"], pparams["Rend"], pparams["nPtrj"]+1)
    q = pplot.compute_q()
    iota = pplot.compute_iota()
    np.savetxt("r-squared.txt", rs)
    np.savetxt("q-squared.txt", q)
    np.savetxt("iota-squared.txt", iota)