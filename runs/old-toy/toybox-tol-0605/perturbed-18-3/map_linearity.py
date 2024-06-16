from pyoculus.problems import AnalyticCylindricalBfield
from pyoculus.solvers import FixedPoint, Manifold
from pyoculus.integrators import RKIntegrator
import matplotlib.pyplot as plt
import numpy as np

import datetime
import pickle
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

log = logging.getLogger(__name__)

def linearized_error(fun, rfp, eigenvector, eps, rtol=1e-10, dt = 2*np.pi):
    """Metric to evaluate if the point rfp + epsilon * eigenvector is in the linear regime of the fixed point."""
    
    iparams = dict()
    iparams["rtol"] = rtol
    iparams["ode"] = fun
    integrator = RKIntegrator(iparams)

    # Initial point and evolution
    rEps = rfp + eps * eigenvector

    ic = np.array([*rEps], dtype=np.float64)
    integrator.set_initial_value(0, ic)
    endpoint1 = integrator.integrate(dt)
    print(endpoint1)

    # Direction of the evolution
    eps_dir = endpoint1 - rfp
    norm_eps_dir = np.linalg.norm(eps_dir)
    eps_dir_norm = eps_dir / norm_eps_dir

    return 1-np.dot(eps_dir_norm, eigenvector)

def linearized_error_jacobian(fun, rtol=1e-10, initpoint=None, vector=None, eps=None, dt = 2*np.pi):
    iparams = dict()
    iparams["rtol"] = rtol
    iparams["ode"] = fun

    integrator = RKIntegrator(iparams)

    if initpoint is None:
        raise ValueError("initpoint is not set")
    if vector is None:
        vector = eps * np.random.random(2)

    ic = np.array([initpoint[0], initpoint[1], 1.0, 0.0, 0.0, 1.0], dtype=np.float64)
    integrator.set_initial_value(0, ic)
    M = integrator.integrate(dt)
    M = M[2:6].reshape((2, 2)).T
    log.info(M)

    inputpoint = initpoint + vector
    ic = np.array([inputpoint[0], inputpoint[1], 1.0, 0.0, 0.0, 1.0], dtype=np.float64)
    integrator.set_initial_value(0, ic)
    endpoint2 = integrator.integrate(dt)[0:2]

    return np.linalg.norm(
        ((M @ vector) - (endpoint2 - initpoint)) / np.linalg.norm(eps)
    )


if __name__ == "__main__":
    fig, ax = plt.subplots()
    # dot = False

    for rtol in [1e-15, 1e-13, 1e-10, 1e-8]:

        separatrix = {"type": "circular-current-loop", "amplitude": -10, "R": 6, "Z": -5.5}
        maxwellboltzmann = {"m": 18, "n": -3, "d": np.sqrt(2), "type": "maxwell-boltzmann", "amplitude": 1e-2}

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

        # # Adding perturbation after the object is created uses the found axis as center point
        pyoproblem.add_perturbation(maxwellboltzmann)

        ### Finding the X-point
        log.info("\nFinding the X-point\n")

        # set up the integrator for the FixedPoint
        iparams = dict()
        iparams["rtol"] = rtol

        pparams = dict()
        pparams["nrestart"] = 0
        pparams["niter"] = 300

        # set up the FixedPoint object
        fixedpoint = FixedPoint(pyoproblem, pparams, integrator_params=iparams)

        # find the X-point
        guess = [6.21560891, -4.46981856]
        log.info(f"Initial guess: {guess}")

        fixedpoint.compute(guess=guess, pp=0, qq=1, sbegin=4, send=9, tol=1e-10)

        if fixedpoint.successful:
            results = [list(p) for p in zip(fixedpoint.x, fixedpoint.y, fixedpoint.z)]
        else:
            raise ValueError("X-point not found")

        iparams = dict()
        iparams["rtol"] = 1e-12

        manifold = Manifold(fixedpoint, pyoproblem, integrator_params=iparams)
        
        # Choose the tangles to work with
        manifold.choose(0, 0)


        ## Compute the linearized error
        epsilon = np.logspace(-20, 1, 200)

        errors_1 = np.zeros(200)
        errors_2 = np.zeros(200)
        # if not dot:
        for i, eps in enumerate(epsilon):
            try:
                errors_1[i] = linearized_error_jacobian(
                    pyoproblem.f_RZ_tangent, rtol=rtol, initpoint=manifold.rfp_u, vector=eps*manifold.vector_u,
                    eps=eps, dt = 2*np.pi
                )
            except:
                errors_1[i] = np.nan
            logging.info(f"eps: {eps}, error: {errors_1[i]}")
        # else:
        for i, eps in enumerate(epsilon):
            try:
                errors_2[i] = linearized_error(
                    pyoproblem.f_RZ, manifold.rfp_u, manifold.vector_u, eps, rtol=rtol, dt=2*np.pi
                )
            except:
                errors_2[i] = np.nan
            logging.info(f"eps: {eps}, error: {errors_2[i]}")

        ax.loglog(epsilon, errors_1, label=f"jac rtol = {rtol}")
        ax.loglog(epsilon, errors_2, label=f"dot rtol = {rtol}")
    
    ### Plotting the results
    # if not dot:
    #     ax.set_title(
    #         f"Linearity of the map at the X point : Jacobian"
    #     )
    # else:
    #     ax.set_title(
    #         f"Linearity of the map at the X point : Dot product"
    #     )
    ax.legend()
    ax.set_xlabel("eps")
    ax.set_ylabel("error")

    plt.show()

    date = datetime.datetime.now().strftime("%m%d%H%M")
    dumpname = f"map_linearity_{date}"
    fig.savefig(dumpname + ".png")
    with open(dumpname + ".pkl", "wb") as f:
        pickle.dump(fig, f)
