from pyoculus.problems.toybox import A_squared
from pyoculus.problems import CylindricalBfield, AnalyticCylindricalBfield
from pyoculus.solvers import PoincarePlot, FixedPoint, Manifold
import matplotlib.pyplot as plt
import numpy as np

ps = AnalyticCylindricalBfield(6, 0, 0.91, 0.6)

# set up the integrator
iparams = dict()
iparams["rtol"] = 1e-7

pparams = dict()
pparams["nrestart"] = 0
pparams['niter'] = 600

fp_perturbed = FixedPoint(ps, pparams, integrator_params=iparams)
fp_perturbed.compute(guess=[6, 0], pp=0, qq=1, sbegin=1, send=8, tol = 1e-10)

iparams = dict()
iparams["rtol"] = 1e-12

manifold = Manifold(fp_perturbed, ps, integrator_params=iparams)
manifold.choose(fp_num_1=0,fp_num_2=0)

# point = np.array([ps._R0, ps._Z0])
point = np.array([ps._R0, ps._Z0])

intA = manifold.integrate_single(point, 1, 1, ret_jacobian=False, integrate_A=True)
expA = A_squared([point[0], 0., point[1]], 6, 0, 0.91, 0.6)

breakpoint()
