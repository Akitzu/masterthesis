import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from pyoculus.problems.toybox import psi_maxwellboltzmann, psitob
from jax import jit

# b = jit(psitob(psi_maxwellboltzmann))

### Creating the pyoculus problem object
from pyoculus.problems import AnalyticCylindricalBfield

separatrix = {"type": "circular-current-loop", "amplitude": -10, "R": 6, "Z": -5.5}
maxwellboltzmann = {"m": 6, "n": -1, "d": np.sqrt(2), "type": "maxwell-boltzmann", "amplitude": 1e-3}

# Creating the pyoculus problem object, adding the perturbation here use the R, Z provided as center point
pyoproblem = AnalyticCylindricalBfield.without_axis(
    6,
    0,
    0.6,
    perturbations_args=[separatrix],
    Rbegin=1,
    Rend=8,
    niter=800,
    guess=[6.41, -0.7],
    tol=1e-9,
)
# # Adding perturbation after the object is created uses the found axis as center point
pyoproblem.add_perturbation(maxwellboltzmann, find_axis=False)

b = pyoproblem.B_perturbations

### Plotting region of conv.

Rs, Zs = np.meshgrid(6+np.concatenate((-np.logspace(-40, -8, 20), np.logspace(-40, -8, 20))), np.linspace(-0.1, 0.1, 11))
Bs = np.array([b([R , 0., Z], 6, 0., 1.41, 3, 1) for R, Z in zip(Rs.flatten(), Zs.flatten())]).reshape(Rs.shape + (3,))
nns = np.isnan(Bs).sum(axis=2)

vmin, vmax = np.percentile(nns, [5, 95])
norm = colors.Normalize(vmin=vmin, vmax=vmax)

mesh = plt.pcolormesh(Rs, Zs, nns, shading='nearest', cmap='viridis', norm=norm)
plt.show()