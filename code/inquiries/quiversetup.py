from pyoculus.problems import AnalyticCylindricalBfield
from pyoculus.solvers import PoincarePlot, FixedPoint, Manifold
import matplotlib.pyplot as plt
import pickle
import numpy as np

### Creating the pyoculus problem object
print("\nCreating the pyoculus problem object\n")

separatrix = {"type": "circular-current-loop", "amplitude": -10, "R": 6, "Z": -5.5}
maxwellboltzmann = {"m": 6, "n": -1, "d": np.sqrt(2), "type": "maxwell-boltzmann", "amplitude": 1e-3}

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

# num_points = 20
# R = np.random.uniform(3.5, 9.2, num_points)
# Z = np.random.uniform(-6, 2.2, num_points)
# RZ_manifold = np.column_stack((R, Z))

R, Z = np.meshgrid(np.linspace(3.5, 9.2, 30), np.linspace(-6, 2.2, 30))
RZ_manifold = np.column_stack((R.flatten(), Z.flatten()))

pyoproblem.plot_intensities([3.5, 9.2], [-6, 2.2], [200, 200], RZ_manifold, N_levels = 100)

plt.show()