from pyoculus.problems import AnalyticCylindricalBfield
from pyoculus.solvers import PoincarePlot, FixedPoint, Manifold
import matplotlib.pyplot as plt
from horus import plot_poincare_pyoculus
plt.style.use('lateky')
import numpy as np
from pathlib import Path

saving_folder = Path('figs').absolute()

pyoproblem = AnalyticCylindricalBfield(
    6,
    0,
    0.8875,
    0.2
)

maxwellboltzmann = {"m": 3, "n": -2, "d": 1.75/np.sqrt(2), "type": "maxwell-boltzmann", "amplitude": 1e-1}
pyoproblem.add_perturbation(maxwellboltzmann)

fig, ax = plt.subplots()
xydata = np.load("data/unperturbed.npy")
plot_poincare_pyoculus(xydata, ax, xlims=None, ylims=None, linewidths=0.3)

ax.set_xlim(3.8, 8.2)
ax.set_ylim(-2.2, 2.2)

rho = 1.75
theta = np.linspace(0, 2*np.pi, 100)
RZs = np.array([[6 + rho*np.cos(t), rho*np.sin(t)] for t in theta])

pyoproblem.plot_intensities(ax = ax, rw=[3.8, 8.2], zw=[-2.2, 2.2], nl=[200, 200], RZ_manifold = RZs, N_levels=200, alpha = 0.5, zorder=11)

fig.savefig("figs/psi_pert.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
fig.savefig("figs/psi_pert.pdf", dpi=300, bbox_inches='tight', pad_inches=0.1)