from pyoculus.problems import AnalyticCylindricalBfield
import matplotlib.pyplot as plt
plt.style.use('lateky')
import numpy as np
from horus import plot_iota_q

Ratio = (9.2-3.5)/(2.5+6)

# ### Creating the pyoculus problem object
# separatrix = {"type": "circular-current-loop", "amplitude": -10, "R": 6, "Z": -5.5}
# # Creating the pyoculus problem object, adding the perturbation here use the R, Z provided as center point
# pyoproblem = AnalyticCylindricalBfield.without_axis(
#     6,
#     0,
#     0.91,
#     0.6,
#     perturbations_args=[separatrix],
#     Rbegin=1,
#     Rend=8,
#     niter=800,
#     guess=[6.41, -0.7],
#     tol=1e-9,
# )

r = np.loadtxt("data/r-squared.txt")
q = np.loadtxt("data/q-squared.txt")
iota = np.loadtxt("data/iota-squared.txt")

fig, ax, axins = plot_iota_q(r, iota, q, r_shift = -6.414097811679057)

fig.show()