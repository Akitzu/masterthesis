from pyoculus.problems import AnalyticCylindricalBfield
from pyoculus.solvers import FixedPoint, Manifold
import matplotlib.pyplot as plt
plt.style.use('lateky')
import numpy as np
from pathlib import Path
import sys

DPI = 600

current_folder = Path('').absolute()
latexplot_folder = Path("../../../latex/images/plots").absolute()
saving_folder = Path("figs").absolute()
sys.path.append(str(latexplot_folder))
from plot_poincare import plot_poincare_pyoculus

### Creating the pyoculus problem object
print("\nCreating the pyoculus problem object\n")

separatrix = {"type": "circular-current-loop", "amplitude": -10, "R": 6, "Z": -5.5}
maxwellboltzmann = {"m": 18, "n": -3, "d": np.sqrt(2), "type": "maxwell-boltzmann", "amplitude": 0.1}

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
guess = [6.21560891, -4.46981856]
print(f"Initial guess: {guess}")

fixedpoint.compute(guess=guess, pp=0, qq=1, sbegin=4, send=9, tol=1e-10)

if fixedpoint.successful:
    results = [list(p) for p in zip(fixedpoint.x, fixedpoint.y, fixedpoint.z)]
else:
    raise ValueError("X-point not found")

# Manifold
iparams = dict()
iparams["rtol"] = 1e-12

manifold = Manifold(pyoproblem, fixedpoint, fixedpoint, integrator_params=iparams)

# Choose the tangles to work with
manifold.choose(signs=[[1, 1], [1, 1]])

# Finding the clinics
i, s_shift = 6, 2
n_s, n_u = i+s_shift, i-s_shift
manifold.onworking = manifold.inner
# Find the first clinic
eps_s_i, eps_u_i = 2e-5, 2e-5
manifold.find_clinic_single(eps_s_i, eps_u_i, n_s=n_s, n_u=n_u)
# Find the second clinic
# manifold.find_clinics(n_points=2)

# Compute the manifold
fig, ax = plt.subplots()
xydata = np.load("poincare_18_3.npy")
plot_poincare_pyoculus(xydata, ax, linewidths=0.1)

ax.set_aspect("equal")
ax.scatter(
        pyoproblem._R0, pyoproblem._Z0, marker="o", edgecolors="black", linewidths=1, zorder=11
    )
ax.scatter(
    results[0][0], results[0][2], marker="X", edgecolors="black", linewidths=1, zorder=11
)

manifold.compute(nintersect = 9, neps = 300,  eps_s=1e-6, eps_u=1e-6, directions='inner')
manifold.plot(ax, zorder=12)

fig.savefig(saving_folder / "manifold.png", dpi=DPI, bbox_inches='tight', pad_inches=0.1)

for i, clinic in enumerate(manifold.onworking["clinics"]):
    eps_s_i, eps_u_i = clinic[1:3]
    n_u = 10
    hu_i = manifold.integrate(manifold.onworking["rfp_u"] + eps_u_i * manifold.onworking["vector_u"], n_u, 1)
    ax.scatter(hu_i[0,:], hu_i[1,:], marker='p', color="royalblue", edgecolor='cyan', zorder=20, label=f'$h_{i+1}$')

fig.savefig(saving_folder / "clinic0.png", dpi=DPI, bbox_inches='tight', pad_inches=0.1)