from pyoculus.problems import AnalyticCylindricalBfield
from pyoculus.integrators import RKIntegrator
import matplotlib.pyplot as plt
import numpy as np

import datetime
import pickle
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def linearized_error(fun, rtol = 1e-10, initpoint = None, vector = None, eps = 1e-5):
    iparams = dict()
    iparams["rtol"] = rtol
    iparams["ode"] = fun

    integrator = RKIntegrator(iparams)

    if initpoint is None:
        raise ValueError("initpoint is not set")
    if vector is None:
        vector = eps*np.random.random(2)

    ic = np.array([initpoint[0], initpoint[1], 1.0, 0.0, 0.0, 1.0], dtype=np.float64)
    integrator.set_initial_value(0, ic)
    M = integrator.integrate(2*np.pi)
    endpoint1 = M[0:2]
    M = M[2:6].reshape((2,2)).T

    inputpoint = initpoint + vector
    ic = np.array([inputpoint[0], inputpoint[1], 1.0, 0.0, 0.0, 1.0], dtype=np.float64)
    integrator.set_initial_value(0, ic)
    endpoint2 = integrator.integrate(2*np.pi)[0:2]

    return np.linalg.norm(((M @ vector) - (endpoint2 - endpoint1))/np.linalg.norm(endpoint2 - endpoint1))

if __name__ == "__main__":

    separatrix = {"type": "circular-current-loop", "amplitude": -4, "R": 3, "Z": -2.2}
    maxwellboltzmann = {"m": 3, "n": -2, "d": 1, "type": "maxwell-boltzmann", "amplitude": 1e-5}
    # gaussian10 = {"m": 1, "n": 0, "d": 1, "type": "gaussian", "amplitude": 0.01}

    # Creating the pyoculus problem object, adding the perturbation here use the R, Z provided as center point
    pyoproblem = AnalyticCylindricalBfield.without_axis(
        3,
        0,
        0.91,
        0.7,
        perturbations_args=[separatrix, maxwellboltzmann],
        Rbegin=1,
        Rend=5,
        niter=800,
        guess=[3.0, -0.1],
        tol=1e-9,
    )


    epsilon = np.logspace(-12, -1, 100)
    errors = np.zeros(100)

    point = [3.1072023810385443, -1.655410284892828]
    v = np.random.random(2)
    for i, eps in enumerate(epsilon):
        v_tmp = eps*v
        errors[i] = linearized_error(pyoproblem.f_RZ_tangent, vector = v_tmp, initpoint=point)
        logging.info(f"eps: {eps}, error: {errors[i]}")

    ### Plotting the results
    fig, ax = plt.subplots()
    ax.set_title(f"Linearity of the map at the point : [{point[0]:.4f}, {point[1]:.4f}]")
    ax.set_xlabel("eps")
    ax.set_ylabel("error")
    ax.loglog(epsilon, errors)

    plt.show()  

    date = datetime.datetime.now().strftime("%m%d%H%M")
    dumpname = f"map_linearity_{date}"
    fig.savefig(dumpname + ".png")
    with open(dumpname + ".pkl", "wb") as f:
        pickle.dump(fig, f)