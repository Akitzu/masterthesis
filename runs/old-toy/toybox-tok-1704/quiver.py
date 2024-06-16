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

    ### Finding the X-point of the unperturbed field
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

    iparams = dict()
    iparams["rtol"] = 1e-12

    manifold_unperturbed = Manifold(fixedpoint, pyoproblem, integrator_params=iparams)
    
    # Choose the tangles to work with
    manifold_unperturbed.choose()

    print("\nComputing the manifold\n")
    manifold_unperturbed.compute(nintersect = 9, neps = 300, epsilon=1e-7)

    ### ADDING PERTURBATION

    maxwellboltzmann = {"m": 7, "n": -1, "d": np.sqrt(2), "type": "maxwell-boltzmann", "amplitude": 0.5}
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

    ### Compute the Poincare plot
    if args.compute_poincare:
        print("\nComputing the Poincare plot")

        # set up the integrator for the Poincare
        iparams = dict()
        iparams["rtol"] = 1e-10

        # set up the Poincare plot
        pparams = dict()
        pparams["nPtrj"] = 10
        pparams["nPpts"] = 100
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

        # Combine the two sets of RZs
        RZs = np.concatenate((RZs_1, RZs_2))

        # Set up the Poincare plot object
        pplot = PoincarePlot(pyoproblem, pparams, integrator_params=iparams)

        # # R-only computation
        # pplot.compute()

        # R-Z computation
        pplot.compute(RZs)

        fig, ax = pplot.plot(marker=".", s=1, color="black")
    else:
        fig = pickle.load(open("../path-to-folder/path-to-file.pkl", "rb"))
        ax = fig.gca()
    
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
    n = np.abs(maxwellboltzmann['n'])
    for i in range(1, 2*n):
        guess_i = [eps_s_1*np.power(manifold.lambda_s, i/(2*n)), eps_u_1*np.power(manifold.lambda_u, i/(2*n))]
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

    print("\nPlotting the quiver\n")
    manifold_unperturbed.plot(ax, directions="u+s+", color="black", linewidth = 0.1)

    out = manifold_unperturbed.unstable['+'].T.flatten()
    each = 2*10
    RZ_manifold = np.array([out[::each], out[1::each]]).T

    pyoproblem.plot_intensities(ax = ax, rw=[3.5, 9.2], zw=[-6, 2.2], nl=[200, 200], RZ_manifold = RZ_manifold, N_levels=100, alpha = 0.5)

    if args.no_save:
        # fig.set_size_inches(10, 6) 
        date = datetime.datetime.now().strftime("%m%d%H%M")
        if args.filename:
            dumpname = args.filename
        else:
            dumpname = f"quiver_{date}"
        with open(dumpname + ".pkl", "wb") as f:
            pickle.dump(fig, f)

    plt.show()
    
    if args.no_save:
        fig.savefig(dumpname + ".png")