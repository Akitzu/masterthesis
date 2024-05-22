from pyoculus.problems import AnalyticCylindricalBfield
from pyoculus.solvers import PoincarePlot, FixedPoint, Manifold
import matplotlib.pyplot as plt
plt.style.use('lateky')
import numpy as np
import datetime
import pickle
from multiprocessing import Pool
import os


def homoclinics(m, n, amplitude):
    separatrix = {"type": "circular-current-loop", "amplitude": -10, "R": 6, "Z": -5.5}
    maxwellboltzmann = {"m": m, "n": n, "d": np.sqrt(2), "type": "maxwell-boltzmann", "amplitude": amplitude}

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
    iparams["rtol"] = 1e-13

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

    # Set up the manifold
    iparams = dict()
    iparams["rtol"] = 1e-13
    manifold = Manifold(fixedpoint, pyoproblem, integrator_params=iparams)

    # Choose the tangles to work with
    manifold.choose(0, 0)
    
    # ### Compute the Poincare plot
    # print("\nComputing the Poincare plot")

    # # set up the integrator for the Poincare
    # iparams = dict()
    # iparams["rtol"] = 1e-10

    # # set up the Poincare plot
    # pparams = dict()
    # pparams["nPtrj"] = 10
    # pparams["nPpts"] = 100
    # pparams["zeta"] = 0

    # # # Set RZs for the normal (R-only) computation
    # # pparams["Rbegin"] = 3.01
    # # pparams["Rend"] = 5.5

    # # Directly setting the RZs
    # # nfieldlines = pparams["nPtrj"] + 1
    # # Rs = np.linspace(3.2, 3.15, nfieldlines)
    # # Zs = np.linspace(-0.43, -2.5, nfieldlines)
    # # RZs = np.array([[r, z] for r, z in zip(Rs, Zs)])

    # # Set RZs for the tweaked (R-Z) computation
    # frac_nf_1 = 2/3
    # nfieldlines_1, nfieldlines_2 = int(np.ceil(frac_nf_1*pparams["nPtrj"])), int(np.floor((1-frac_nf_1)*pparams["nPtrj"]))+1

    # # Two interval computation opoint to xpoint then xpoint to coilpoint
    # xpoint = np.array([results[0][0], results[0][2]])
    # opoint = np.array([pyoproblem._R0, pyoproblem._Z0])
    # coilpoint = np.array(
    #     [pyoproblem.perturbations_args[0]["R"], pyoproblem.perturbations_args[0]["Z"]]
    # )

    # # Sophisticated way more around the xpoint
    # deps = 0.05
    # frac_n1 = 1/2
    # n1, n2 = int(np.ceil(frac_n1 * nfieldlines_2)), int(np.floor((1 - frac_n1) * nfieldlines_2))
    # RZ1 = xpoint + deps * (1 - np.linspace(0, 1, n1)).reshape((n1, 1)) @ (
    #     opoint - xpoint
    # ).reshape((1, 2))
    # RZ2 = xpoint + deps * np.linspace(0, 1, n2).reshape((n2, 1)) @ (
    #     coilpoint - xpoint
    # ).reshape((1, 2))
    # RZs_2 = np.concatenate((RZ1, RZ2))

    # # Simple way from opoint to xpoint then to coilpoint
    # frac_n1 = 3/4
    # n1, n2 = int(np.ceil(frac_n1 * nfieldlines_1)), int(np.floor((1 - frac_n1) * nfieldlines_1))
    # Rs = np.concatenate((np.linspace(opoint[0]+1e-4, xpoint[0]-deps, n1), np.linspace(xpoint[0]+deps, coilpoint[0]-1e-4, n2)))
    # Zs = np.concatenate((np.linspace(opoint[1]+1e-4, xpoint[1]-deps, n1), np.linspace(xpoint[1]+deps, coilpoint[1]-1e-4, n2)))
    # RZs_1 = np.array([[r, z] for r, z in zip(Rs, Zs)])
    
    # # Combine the two sets of RZs
    # RZs = np.concatenate((RZs_1, RZs_2))

    # # Set up the Poincare plot object
    # pplot = PoincarePlot(pyoproblem, pparams, integrator_params=iparams)

    # # # R-only computation
    # # pplot.compute()

    # # R-Z computation
    # pplot.compute(RZs)

    # fig, ax = pplot.plot(marker=".", s=1, color="black")
    
    # ax.scatter(
    #         pyoproblem._R0, pyoproblem._Z0, marker="o", edgecolors="black", linewidths=1
    #     )
    # ax.scatter(
    #     results[0][0], results[0][2], marker="X", edgecolors="black", linewidths=1
    # )

    # # compute manifold
    # ax.set_title(f"amplitude = {maxwellboltzmann['amplitude']}, m = {maxwellboltzmann['m']}, n = {maxwellboltzmann['n']}, d = {maxwellboltzmann['d']:.2f}")

    print("\nFinding homoclinics\n")
    manifold.find_clinics(n_points=6)
    if len(manifold.clinics) != 6:
        raise ValueError("Not able to find all the clinics")

    # n_s = manifold.find_clinic_configuration['n_s'] - 1
    # n_u = manifold.find_clinic_configuration['n_u']

    # manifold.compute(neps=100, nintersect= n_s + n_u)
    # manifold.plot(ax, directions="u+s+")
    
    # marker = ["X", "o", "s", "p", "P", "*", "x", "D", "d", "^", "v", "<", ">"]
    # for i, clinic in enumerate(manifold.clinics):
    #     _, eps_u_i = clinic[1:3]
        
    #     # hs_i = manifold.integrate(manifold.rfp_s + eps_s_i * manifold.vector_s, n_s, -1)
    #     hu_i = manifold.integrate(manifold.rfp_u + eps_u_i * manifold.vector_u, n_u + n_s, 1)
    #     # ax.scatter(hs_i[0,:], hs_i[1,:], marker=marker[i], color="royalblue", edgecolor='cyan', zorder=10)
    #     ax.scatter(hu_i[0,:], hu_i[1,:], marker=marker[i], color="royalblue", edgecolor='cyan', zorder=10, label=f'$h_{i+1}$')

    # fig.set_size_inches(12, 12)
    # ax.set_xlim(3.5, 9.2)
    # ax.set_ylim(-6, 2.5)
    # ax.set_xlabel(r'R [m]')
    # ax.set_ylabel(r'Z [m]')

    # date = datetime.datetime.now().strftime("%m%d%H%M")
    # dumpname = f"area_{m}_{n}_{amplitude:.3e}_{date}"

    # os.makedirs("figures", exist_ok=True)
    # with open("figures/" + dumpname + ".pkl", "wb") as f:
    #     pickle.dump(fig, f)
    
    # fig.savefig("figures/" + dumpname + ".png")
    # fig, ax = None, None
    # plt.close()

    manifold.resonance_area()
    return manifold.areas

# Define a function to be run in each process
def process_func(args):
    m, n = 12, -2
    amplitude = args
    print(f"Running for m = {m}, n = {n}, amplitude = {amplitude:.5f}")
    try:
        res = homoclinics(m, n, amplitude)
        return [amplitude, res]
    except Exception as e:
        print(e)
        return []


if __name__ == "__main__":
    amp = np.linspace(1e-2, 1, 4)
    # Create a pool of 4 processes
    with Pool(4) as p:
        results = p.map(process_func, amp)

    pickle.dump(results, open("areas.pkl", "wb"))
