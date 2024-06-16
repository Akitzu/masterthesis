from pyoculus.problems import AnalyticCylindricalBfield
from pyoculus.solvers import FixedPoint, Manifold
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append("..")
from plot_poincare import plot_poincare_pyoculus

### Creating the pyoculus problem object
print("\nCreating the pyoculus problem object\n")

separatrix = {"type": "circular-current-loop", "amplitude": -10, "R": 6, "Z": -5.5}
maxwellboltzmann = {"m": 6, "n": -1, "d": np.sqrt(2), "type": "maxwell-boltzmann", "amplitude": 0.1}

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

### Manifold computation
iparams = dict()
iparams["rtol"] = 1e-12

manifold_unperturbed = Manifold(pyoproblem, fixedpoint, fixedpoint, integrator_params=iparams)

# Choose the tangles to work with
manifold_unperturbed.choose(signs=[[1, 1], [1, 1]])

print("\nComputing the manifold\n")
eps_s, eps_u = 1e-6, 1e-6
manifold_unperturbed.compute(nintersect = 9, neps = 50,  eps_s=eps_s, eps_u=eps_u, directions='inner')

out = manifold_unperturbed.inner["lfs"]["unstable"].T.flatten()
each = 2*2
RZ_manifold = np.array([out[::each], out[1::each]]).T

# # Adding perturbation after the object is created uses the found axis as center point
pyoproblem.add_perturbation(maxwellboltzmann)

xydata = np.load("../toytok/poincare_cleaner.npy")
fig, ax = plt.subplots()
plot_poincare_pyoculus(xydata, ax, linewidths=0.1, zorder=10)

pyoproblem.plot_intensities(ax = ax, rw=[3.5, 9.2], zw=[-6, 2.5], nl=[200, 200], RZ_manifold = RZ_manifold, N_levels=200, alpha = 0.5, zorder=11)

# fig.savefig("perturbation.png", dpi=300, bbox_inches="tight", pad_inches=0.1) 
# fig.savefig("perturbation.pdf", dpi=300, bbox_inches="tight", pad_inches=0.1)