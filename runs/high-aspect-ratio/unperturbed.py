from pyoculus.problems import AnalyticCylindricalBfield
from pyoculus.solvers import PoincarePlot
import matplotlib.pyplot as plt
plt.style.use('lateky')
import numpy as np

# Creating the pyoculus problem object, adding the perturbation here use the R, Z provided as center point
pyoproblem = AnalyticCylindricalBfield(
    6,
    0,
    0.8875,
    0.2
)

# set up the integrator for the Poincare
iparams = dict()
iparams["rtol"] = 1e-10

# set up the Poincare plot
pparams = dict()
pparams["nPtrj"] = 30
pparams["nPpts"] = 400
pparams["zeta"] = 0

# Set RZs for the normal (R-only) computation   
pparams["Rbegin"] = pyoproblem._R0+1e-3
pparams["Rend"] = 8

# Set up the Poincare plot object
pplot = PoincarePlot(pyoproblem, pparams, integrator_params=iparams)

# # R-only computation
pplot.compute()
pplot.save("data/unperturbed.npy")

# Iota plot
rs = np.linspace(pparams["Rbegin"], pparams["Rend"], pparams["nPtrj"]+1)
q = pplot.compute_q()
iota = pplot.compute_iota()
np.savetxt("data/r-squared.txt", rs)
np.savetxt("data/q-squared.txt", q)
np.savetxt("data/iota-squared.txt", iota)