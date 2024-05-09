from pyoculus.problems import CylindricalBfield, AnalyticCylindricalBfield
from pyoculus.solvers import PoincarePlot, FixedPoint, Manifold
from pyoculus.integrators import RKIntegrator
import matplotlib.pyplot as plt
import numpy as np
import datetime
import pickle

# separatrix = {"type": "circular-current-loop", "amplitude": -4.2, "R": 3, "Z": -2.2}
separatrix = {"type": "circular-current-loop", "amplitude": -10, "R": 6, "Z": -5.5}
# separatrix = {"type": "circular-current-loop", "amplitude": -4, "R": 3, "Z": -2.2}
maxwellboltzmann = {"m": 7, "n": -1, "d": np.sqrt(2), "type": "maxwell-boltzmann", "amplitude": 0.1, "A": 1, "B": 2}
# gaussian10 = {"m": 1, "n": 0, "d": 1, "type": "gaussian", "amplitude": 0.1}

ps = AnalyticCylindricalBfield.without_axis(6, 0, 0.91, 0.6, perturbations_args = [separatrix], Rbegin = 2, Rend = 8, niter = 800, guess=[6.4,-0.7],  tol = 1e-9)
ps.add_perturbation(maxwellboltzmann)
# ps = AnalyticCylindricalBfield(3, 0, 0.9, 0.7, perturbations_args = [separatrix])

iparams = dict()
iparams["rtol"] = 1e-20
iparams["ode"] = ps.f_RZ_tangent
iparams["type"] = "dop853"

integrator = RKIntegrator(iparams)

eps_s = 1.3848672717825134e-08
eps_u = 6.057629111233517e-08
rfp = np.array([6.205718761242266,-4.496620141231264])
vector_s = np.array([-0.49822014,  0.86705057])
vector_u = np.array([0.74535851, 0.66666385])

# initpoint = rfp + eps_s*vector_s
initpoint = rfp + eps_u*vector_u


def integrate(n):
    ic = np.array([initpoint[0], initpoint[1], 1., 0., 0., 1.], dtype=np.float64)
    integrator.set_initial_value(0, ic)
    return integrator.integrate(n*2*np.pi)

integrate(5)
breakpoint()
integrator.integrate(6*2*np.pi)