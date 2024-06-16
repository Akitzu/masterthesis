from pyoculus.problems import AnalyticCylindricalBfield
from pyoculus.solvers import PoincarePlot, FixedPoint, Manifold
import matplotlib.pyplot as plt
plt.style.use('lateky')

import numpy as np

# Creating the pyoculus problem object, adding the perturbation here use the R, Z provided as center point
pyoproblem = AnalyticCylindricalBfield(
    6,
    0,
    1,
    0.5
)

# set up the integrator for the Poincare
iparams = dict()
iparams["rtol"] = 1e-10

# set up the Poincare plot
pparams = dict()
pparams["nPtrj"] = 10
pparams["nPpts"] = 300
pparams["zeta"] = 0

# Set RZs for the normal (R-only) computation   
pparams["Rbegin"] = pyoproblem._R0+1e-3
pparams["Rend"] = 8

# Set up the Poincare plot object
pplot = PoincarePlot(pyoproblem, pparams, integrator_params=iparams)

# # R-only computation
pplot.compute()
pplot.save("poincare.npy")

# Iota plot
rs = np.linspace(pparams["Rbegin"], pparams["Rend"], pparams["nPtrj"]+1)
q = pplot.compute_q()
iota = pplot.compute_iota()
np.savetxt("r-squared.txt", rs)
np.savetxt("q-squared.txt", q)
np.savetxt("iota-squared.txt", iota)

