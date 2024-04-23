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
    maxwellboltzmann = {"m": 13, "n": -2, "d": np.sqrt(2), "type": "maxwell-boltzmann", "amplitude": 1e-7}

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

    ### Compute the Poincare plot
    if args.compute_poincare:
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
        frac_nf_1 = 1/3
        nfieldlines_1, nfieldlines_2 = int(np.ceil(frac_nf_1*pparams["nPtrj"])), int(np.floor((1-frac_nf_1)*pparams["nPtrj"]))+1

        # Two interval computation opoint to xpoint then xpoint to coilpoint
        frac_n1 = 3/4
        n1, n2 = int(np.ceil(frac_n1 * nfieldlines_1)), int(np.floor((1 - frac_n1) * nfieldlines_1))
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
    
    guess_2 = [eps_s_1*np.power(manifold.lambda_s, 1/4), eps_u_1*np.power(manifold.lambda_u, 1/4)]
    print(f"2nd initial guess: {guess_2}")   
    eps_s_2, eps_u_2 = manifold.find_homoclinic(guess_2[0], guess_2[1], n_s = 7, n_u = 6)

    guess_3 = [eps_s_1*np.power(manifold.lambda_s, 2/4), eps_u_1*np.power(manifold.lambda_u, 2/4)]     
    print(f"3rd initial guess: {guess_3}")   
    eps_s_3, eps_u_3 = manifold.find_homoclinic(guess_3[0], guess_3[1], n_s = 7, n_u = 6) 

    guess_4 = [eps_s_1*np.power(manifold.lambda_s, 3/4), eps_u_1*np.power(manifold.lambda_u, 3/4)]     
    print(f"4rth initial guess: {guess_4}") 
    eps_s_4, eps_u_4 = manifold.find_homoclinic(guess_4[0], guess_4[1], n_s = 7, n_u = 6)

    # Plot the homoclinic points
    print("\nPlotting homoclinic points")
    hs_1 = manifold.integrate(manifold.rfp_s + eps_s_1 * manifold.vector_s, 7, -1)
    hs_2 = manifold.integrate(manifold.rfp_s + eps_s_2 * manifold.vector_s, 7, -1)
    hs_3 = manifold.integrate(manifold.rfp_s + eps_s_3 * manifold.vector_s, 7, -1)
    hs_4 = manifold.integrate(manifold.rfp_s + eps_s_4 * manifold.vector_s, 7, -1)

    hu_1 = manifold.integrate(manifold.rfp_u + eps_u_1 * manifold.vector_u, 6, 1)
    hu_2 = manifold.integrate(manifold.rfp_u + eps_u_2 * manifold.vector_u, 6, 1)
    hu_3 = manifold.integrate(manifold.rfp_u + eps_u_3 * manifold.vector_u, 6, 1)
    hu_4 = manifold.integrate(manifold.rfp_u + eps_u_4 * manifold.vector_u, 6, 1)

    ax.scatter(hs_1[0,:], hs_1[1,:], marker="x", color="purple", zorder=10)
    ax.scatter(hs_2[0,:], hs_2[1,:], marker="+", color="purple", zorder=10)
    ax.scatter(hs_3[0,:], hs_3[1,:], marker="o", color="purple", zorder=10)
    ax.scatter(hs_4[0,:], hs_4[1,:], marker="s", color="purple", zorder=10)

    ax.scatter(hu_1[0,:], hu_1[1,:], marker="x", color="blue", zorder=10)
    ax.scatter(hu_2[0,:], hu_2[1,:], marker="+", color="blue", zorder=10)
    ax.scatter(hu_3[0,:], hu_3[1,:], marker="o", color="blue", zorder=10)
    ax.scatter(hu_4[0,:], hu_4[1,:], marker="s", color="blue", zorder=10)

    print("\nComputing the manifold\n")
    manifold.compute(nintersect = 9, neps = 300, epsilon=1e-7)

    print("\nPlotting the manifold\n")
    manifold.plot(ax, directions="u+s+")
    ax.set_title(f"amplitude = {maxwellboltzmann['amplitude']}, m = {maxwellboltzmann['m']}, n = {maxwellboltzmann['n']}, d = {maxwellboltzmann['d']:.2f}")

    if args.no_save:
        # fig.set_size_inches(10, 6) 
        date = datetime.datetime.now().strftime("%m%d%H%M")
        if args.filename:
            dumpname = args.filename
        else:
            dumpname = f"homoclinics_{date}"
        with open(dumpname + ".pkl", "wb") as f:
            pickle.dump(fig, f)

    plt.show()
    
    if args.no_save:
        fig.savefig(dumpname + ".png")