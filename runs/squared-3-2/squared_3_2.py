from pyoculus.problems import AnalyticCylindricalBfield
from pyoculus.solvers import PoincarePlot, FixedPoint, Manifold
import matplotlib.pyplot as plt
plt.style.use('lateky')
import numpy as np
from pathlib import Path
import sys

DPI = 600

current_folder = Path('').absolute()
latexplot_folder = Path("../../latex/images/plots").absolute()
sys.path.append(str(latexplot_folder))
from plot_poincare import plot_poincare_pyoculus

pyoproblem = AnalyticCylindricalBfield(
    6,
    0,
    0.8875,
    0.2
)

maxwellboltzmann = {"m": 3, "n": -2, "d": 1.75/np.sqrt(2), "type": "maxwell-boltzmann", "amplitude": 1e-1}
pyoproblem.add_perturbation(maxwellboltzmann)

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
pparams["Rend"] = 8.5

# Set up the Poincare plot object
pplot_perturbed = PoincarePlot(pyoproblem, pparams, integrator_params=iparams)

# # R-only computation
# pplot_perturbed.compute()

# fig, ax = pplot_perturbed.plot(marker=".", s=1)
pplot_perturbed.save("poincare.npy")

fig, ax = plt.subplots()
xydata = np.load("poincare.npy")
plot_poincare_pyoculus(xydata, ax, xlims=None, ylims=None, linewidths=0.3)

fig.savefig("figs/poincare_plot.png", dpi=300, bbox_inches='tight', pad_inches=0.1)

guess = [4.2, 0.]
# set up the integrator for the FixedPoint
iparams = dict()
iparams["rtol"] = 1e-13

pparams = dict()
pparams["nrestart"] = 0
pparams["niter"] = 300
pparams['Z'] = 0

# set up the FixedPoint object
fixedpoint = FixedPoint(pyoproblem, pparams, integrator_params=iparams)

# find the X-point
fixedpoint.compute(guess=guess, pp=2, qq=3, sbegin=2, send=10, tol=1e-10)

if fixedpoint.successful:
    results = [list(p) for p in zip(fixedpoint.x, fixedpoint.y, fixedpoint.z)]
else:
    raise ValueError("X-point not found")

pparams.pop("Z")
# set up the FixedPoint object
fp_x2 = FixedPoint(pyoproblem, pparams, integrator_params=iparams)
fp_x2.compute(guess=[6.44042536414122, 1.7065049712562115], pp=2, qq=3, sbegin=2, send=10, tol=1e-10)

for rr in results:
    ax.scatter(rr[0], rr[2], marker="X", edgecolors="black", linewidths=1)
fig.savefig("figs/fixedpoints.png", dpi=300, bbox_inches='tight', pad_inches=0.1)

iparams = dict()
iparams["rtol"] = 1e-13
manifold = Manifold(pyoproblem, fixedpoint, fp_x2, integrator_params=iparams)

manifold.choose(signs=[[1, -1], [-1, 1]])

manifold.compute(neps=30, nintersect=8, directions="inner")
manifold.plot(ax=ax, directions="isiu")
fig.savefig("figs/inner.png", dpi=300, bbox_inches='tight', pad_inches=0.1)

manifold.compute(neps=30, nintersect=9, directions="outer")
manifold.plot(ax=ax, directions="osou")
fig.savefig("figs/outer.png", dpi=300, bbox_inches='tight', pad_inches=0.1)

manifold.onworking = manifold.outer
manifold.find_clinics(n_points=4)

manifold.onworking = manifold.inner
manifold.find_clinics(n_points=4)

manifold.onworking = manifold.outer
marker = ["s", "p", "P", "*", "x", "D", "d", "^", "v", "<", ">"]
confns = manifold.onworking["find_clinic_configuration"]
n_u = confns["n_u"]+confns["n_s"]+2

manifold.onworking = manifold.inner
manifold.turnstile_area()
for i, clinic in enumerate(manifold.onworking["clinics"]):
    eps_s_i, eps_u_i = clinic[1:3]
    
    hu_i = manifold.integrate(manifold.onworking["rfp_u"] + eps_u_i * manifold.onworking["vector_u"], n_u, 1)
    ax.scatter(hu_i[0,:], hu_i[1,:], marker=marker[i], color="royalblue", edgecolor='cyan', zorder=10, label=f'$h_{i+1}$')
    fig.savefig(f"figs/heteroclinics_inner_{i}.png", dpi=300, bbox_inches='tight', pad_inches=0.1)

manifold.onworking = manifold.outer
manifold.turnstile_area()
for i, clinic in enumerate(manifold.onworking["clinics"]):
    eps_s_i, eps_u_i = clinic[1:3]
    
    hu_i = manifold.integrate(manifold.onworking["rfp_u"] + eps_u_i * manifold.onworking["vector_u"], n_u, 1)
    ax.scatter(hu_i[0,:], hu_i[1,:], marker=marker[i], color="royalblue", edgecolor='cyan', zorder=10, label=f'$h_{i+1}$')
    fig.savefig(f"figs/heteroclinics_outer_{i}.png", dpi=300, bbox_inches='tight', pad_inches=0.1)

ax.set_xlim(5.2, 6.5)
ax.set_ylim(1.4, 2.1)
fig.savefig("figs/closeup.png", dpi=300, bbox_inches='tight', pad_inches=0.1)

inner_areas = manifold.inner["areas"][:,0]
outer_areas = manifold.outer["areas"][:,0]

import pandas as pd
data = [
    {"type": "inner", "area": inner_areas[inner_areas > 0].sum(), "Error_by_diff": manifold.inner["areas"][:, 1][inner_areas > 0].sum(), "Error_by_estim": manifold.inner["areas"][:, 2][inner_areas > 0].sum(), "total_sum": inner_areas.sum()},
    {"type": "outer", "area": outer_areas[outer_areas > 0].sum(), "Error_by_diff": manifold.outer["areas"][:, 1][outer_areas > 0].sum(), "Error_by_estim": manifold.outer["areas"][:, 2][outer_areas > 0].sum(), "total_sum": outer_areas.sum()},
]

df = pd.DataFrame(data)
df.to_csv("areas.csv")