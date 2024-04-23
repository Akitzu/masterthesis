from pyoculus.problems import AnalyticCylindricalBfield
from pyoculus.solvers import PoincarePlot, FixedPoint, Manifold
import matplotlib.pyplot as plt
import pickle
import numpy as np

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
# guess = [6.18, -4.49]
guess = [6.21560891, -4.46981856]
print(f"Initial guess: {guess}")

fixedpoint.compute(guess=guess, pp=0, qq=1, sbegin=4, send=9, tol=1e-10)

if fixedpoint.successful:
    results = [list(p) for p in zip(fixedpoint.x, fixedpoint.y, fixedpoint.z)]
else:
    print("FixedPoint did not converge - don't continue")

iparams = dict()
iparams["rtol"] = 1e-12

manifold = Manifold(fixedpoint, pyoproblem, integrator_params=iparams)

manifold.choose()

try:
    manifold.find_homoclinic(1e-7, 1e-6, n_s = 8, n_u = 6)
except:
    breakpoint()

fig = pickle.load(open("../../runs/toybox-tok-1704/perturbed-13-2/poincare_04181732.pkl", "rb"))
ax = fig.gca()
ax.set_xlim(3.5, 9.2)

print("\nComputing the manifold\n")
manifold.compute(nintersect = 9, neps = 300, epsilon=1e-7)

print("\nPlotting the manifold")
manifold.plot(ax, directions="u+s+")

print("\nPlotting homoclinic points")
ax.scatter(h1_1[0,:], h1_1[1,:], marker="x", color="purple", zorder=10)
ax.scatter(h1_2[0,:], h1_2[1,:], marker="+", color="purple", zorder=10)
ax.scatter(h2_1[0,:], h2_1[1,:], marker="x", color="black", zorder=10)
ax.scatter(h2_2[0,:], h2_2[1,:], marker="+", color="black", zorder=10)

plt.show()
breakpoint()