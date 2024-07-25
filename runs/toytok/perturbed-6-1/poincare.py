from pyoculus.problems import AnalyticCylindricalBfield
from pyoculus.solvers import PoincarePlot, FixedPoint
import matplotlib.pyplot as plt
import numpy as np
import pickle
import datetime
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute the Poincare plot of a perturbed tokamak field")
    parser.add_argument('--save', type=bool, default=True, help='Saving the plot')
    args = parser.parse_args()

    ### Creating the pyoculus problem object
    print("\nCreating the pyoculus problem object\n")

    separatrix = {"type": "circular-current-loop", "amplitude": -10, "R": 6, "Z": -5.5}
    maxwellboltzmann = {"m": 6, "n": -1, "d": np.sqrt(2), "type": "maxwell-boltzmann", "amplitude": 0.4}

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
    Baxis = np.linalg.norm(pyoproblem.B([pyoproblem._R0, 0., pyoproblem._Z0]) * np.array([1, pyoproblem._R0, 1]))
    # # Adding perturbation after the object is created uses the found axis as center point
    pyoproblem.add_perturbation(maxwellboltzmann)
    Bpert = 0
    for rho in np.linspace(0, pyoproblem._R0, 100):
        for theta in np.linspace(0, 2*np.pi, 100):
            Btmp = np.linalg.norm(pyoproblem.perturbations[1]([pyoproblem._R0+rho*np.cos(theta), 0., pyoproblem._Z0+rho*np.sin(theta)]))
            if Btmp > Bpert:
                Bpert = Btmp
    breakpoint()

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
    # guess = [6.21560891, -4.46981856]
    print(f"Initial guess: {guess}")

    fp.compute(guess=guess, pp=0, qq=1, sbegin=1, send=8, tol=1e-10)

    if fp.successful:
        results = [list(p) for p in zip(fp.x, fp.y, fp.z)]
    else:
        results = [[6.14, 0., -4.45]]

    ### Compute the Poincare plot
    print("\nComputing the Poincare plot\n")

    # set up the integrator for the Poincare
    iparams = dict()
    iparams["rtol"] = 1e-10

    # set up the Poincare plot
    pparams = dict()
    pparams["nPtrj"] = 60
    pparams["nPpts"] = 400
    pparams["zeta"] = 0

    # # Set RZs for the normal (R-only) computation
    # pparams["Rbegin"] = pyoproblem._R0+1e-3
    # pparams["Rend"] = 8.2

    # Set RZs for the tweaked (R-Z) computation
    frac_nf_0 = 0.96
    nfieldlines, nfieldlines_3 = int(np.ceil(frac_nf_0*pparams["nPtrj"])), int(np.floor((1-frac_nf_0)*pparams["nPtrj"]))+1
    
    frac_nf_1 = 2/3
    nfieldlines_1, nfieldlines_2 = int(np.ceil(frac_nf_1*nfieldlines)), int(np.floor((1-frac_nf_1)*nfieldlines))

    # Two interval computation opoint to xpoint then xpoint to coilpoint
    frac_n1 = 3/4
    n1, n2 = int(np.ceil(frac_n1 * nfieldlines_1)), int(np.floor((1 - frac_n1) * nfieldlines_1))
    xpoint = np.array([results[0][0], results[0][2]])
    opoint = np.array([pyoproblem._R0, pyoproblem._Z0])
    coilpoint = np.array(
        [pyoproblem.perturbations_args[0]["R"], pyoproblem.perturbations_args[0]["Z"]]
    )

    # Simple way from opoint to xpoint then to coilpoint
    Rs = np.concatenate((np.linspace(opoint[0]+1e-4, xpoint[0], n1), np.linspace(xpoint[0], coilpoint[0]-1e-4, n2)))
    Zs = np.concatenate((np.linspace(opoint[1]+1e-4, xpoint[1], n1), np.linspace(xpoint[1], coilpoint[1]-1e-4, n2)))
    RZs_1 = np.array([[r, z] for r, z in zip(Rs, Zs)])

    # Sophisticated way more around the xpoint
    frac_n1 = 1/2
    n1, n2 = int(np.ceil(frac_n1 * nfieldlines_2)), int(np.floor((1 - frac_n1) * nfieldlines_2))
    deps = 0.05
    RZ1 = xpoint + deps * (1 - np.linspace(0, 1, n1)).reshape((n1, 1)) @ (
        opoint - xpoint
    ).reshape((1, 2))
    RZ2 = xpoint + deps * np.linspace(0, 1, n2).reshape((n2, 1)) @ (
        coilpoint - xpoint
    ).reshape((1, 2))
    RZs_2 = np.concatenate((RZ1, RZ2))

    # Third interval
    Rs = np.linspace(xpoint[0]+0.1, 8, nfieldlines_3)
    Zs = np.linspace(xpoint[1]-0.1, -5, nfieldlines_3)
    RZs_3 = np.array([[r, z] for r, z in zip(Rs, Zs)])

    RZs = np.concatenate((RZs_1, RZs_2, RZs_3))

    # Set up the Poincare plot object
    pplot = PoincarePlot(pyoproblem, pparams, integrator_params=iparams)

    # # R-only computation
    # pplot.compute()

    # R-Z computation
    pplot.compute(RZs)

    ### Plotting the results
    pplot.save("perturbed_6_1_4e-1.npy")