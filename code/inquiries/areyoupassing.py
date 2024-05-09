from pyoculus.problems import CylindricalBfield, AnalyticCylindricalBfield
from pyoculus.solvers import PoincarePlot, FixedPoint, Manifold
import matplotlib.pyplot as plt
import numpy as np
import datetime
import pickle

# separatrix = {"type": "circular-current-loop", "amplitude": -4.2, "R": 3, "Z": -2.2}
separatrix = {"type": "circular-current-loop", "amplitude": -10, "R": 6, "Z": -5.5}
# separatrix = {"type": "circular-current-loop", "amplitude": -4, "R": 3, "Z": -2.2}
maxwellboltzmann = {"m": 7, "n": -1, "d": np.sqrt(2), "type": "maxwell-boltzmann", "amplitude": 0.1, "A": 1, "B": 2}
# gaussian10 = {"m": 1, "n": 0, "d": 1, "type": "gaussian", "amplitude": 0.1}

ps = AnalyticCylindricalBfield.without_axis(6, 0, 0.91, 0.6, perturbations_args = [separatrix], Rbegin = 2, Rend = 8, niter = 800, guess=[6.4,-0.7],  tol = 1e-9)
ps.add_perturbation(maxwellboltzmann)
# ps = AnalyticCylindricalBfield(3, 0, 0.9, 0.7, perturbations_args = [separatrix])

# set up the integrator
iparams = dict()
iparams["rtol"] = 1e-20

pparams = dict()
pparams["nrestart"] = 0
pparams['niter'] = 600

fp_perturbed = FixedPoint(ps, pparams, integrator_params=iparams)

# fp_perturbed.compute(guess=[fp.x[0], fp.z[0]], pp=0, qq=1, sbegin=0.1, send=6, tol = 1e-10)
# fp_perturbed.compute(guess=[3.117263523069049, -1.6173346133145015], pp=0, qq=1, sbegin=0.1, send=6, tol = 1e-10)
# fp_perturbed.compute(guess=[3.1072023810385443, -1.655410284892828], pp=0, qq=1, sbegin=0.1, send=6, tol = 4e-12)
# fp_perturbed.compute(guess=[3.117264916246293, -1.617334822348791], pp=0, qq=1, sbegin=0.1, send=6, tol = 1e-10)
# fp_perturbed.compute(guess=[4.624454, 0.], pp=0, qq=1, sbegin=0.1, send=6, tol = 1e-10)
# fp_perturbed.compute(guess=[4.43582958 -1.22440153], pp=0, qq=1, sbegin=0.1, send=6, tol = 1e-10)
fp_perturbed.compute(guess=[6.2, -4.45], pp=0, qq=1, sbegin=1, send=8, tol = 1e-10)

results = [list(p) for p in zip(fp_perturbed.x, fp_perturbed.y, fp_perturbed.z)]

# set up the integrator for the Poincare
iparams = dict()
iparams["rtol"] = 1e-10

# set up the Poincare plot
pparams = dict()
pparams["nPtrj"] = 8
pparams["nPpts"] = 150
pparams["zeta"] = 0

# # Set RZs for the normal (R-only) computation
# pparams["Rbegin"] = 6.3
# pparams["Rend"] = 9.1

# Set RZs for the tweaked (R-Z) computation
nfieldlines = pparams["nPtrj"] + 1

# Directly setting the RZs
# Rs = np.linspace(6, 3.15, nfieldlines)
# Zs = np.linspace(-0.43, -2.5, nfieldlines)
# RZs = np.array([[r, z] for r, z in zip(Rs, Zs)])

# Two interval computation opoint to xpoint then xpoint to coilpoint
n1, n2 = int(np.ceil(nfieldlines / 2)), int(np.floor(nfieldlines / 2))
xpoint = np.array([results[0][0], results[0][2]])
opoint = np.array([ps._R0, ps._Z0])
coilpoint = np.array(
    [ps.perturbations_args[0]["R"], ps.perturbations_args[0]["Z"]]
)

# Simple way from opoint to xpoint then to coilpoint
Rs = np.concatenate((np.linspace(opoint[0]+1e-4, xpoint[0], n1), np.linspace(xpoint[0], coilpoint[0]-1e-4, n2)))
Zs = np.concatenate((np.linspace(opoint[1]+1e-4, xpoint[1], n1), np.linspace(xpoint[1], coilpoint[1]-1e-4, n2)))
RZs = np.array([[r, z] for r, z in zip(Rs, Zs)])

# Sophisticated way more around the xpoint
# deps = 0.05
# RZ1 = xpoint + deps * (1 - np.linspace(0, 1, n1)).reshape((n1, 1)) @ (
#     opoint - xpoint
# ).reshape((1, 2))
# RZ2 = xpoint + deps * np.linspace(0, 1, n2).reshape((n2, 1)) @ (
#     coilpoint - xpoint
# ).reshape((1, 2))
# RZs = np.concatenate((RZ1, RZ2))

# Set up the Poincare plot object
pplot = PoincarePlot(ps, pparams, integrator_params=iparams)

# # R-only computation
# pplot.compute()

# R-Z computation
pplot.compute(RZs)

fig_perturbed, ax_perturbed = pplot.plot(marker=".", s=1)

ax_perturbed.scatter(results[0][0], results[0][2], marker="X", edgecolors="black", linewidths=1)
ax_perturbed.scatter(ps._R0, ps._Z0, marker="o", edgecolors="black", linewidths=1)

iparams = dict()
iparams["rtol"] = 1e-20

manifold = Manifold(fp_perturbed, ps, integrator_params=iparams)
manifold.choose(fp_num_1=0,fp_num_2=0)

# Find the homoclinic points
eps_s_1, eps_u_1 = manifold.find_homoclinic(1e-7, 1e-7)

# ev_h1 = manifold.rfp_u + manifold.clinics[0][2] * manifold.vector_u
# ev_h2 = manifold.rfp_s + manifold.clinics[0][1] * manifold.vector_s

ev_h1 = manifold.rfp_u + manifold.clinics[0][2] * manifold.vector_u
ev_h2 = manifold.rfp_u + manifold.clinics[0][2] * manifold.vector_u

# ev_h1 = manifold.clinics[0][0]
# ev_h2 = manifold.clinics[0][0]

for i in range(10):
    if i > 0:
        ev_h1 = manifold.integrate_single(ev_h1, 1, 1, ret_jacobian=False)
        ev_h2 = manifold.integrate_single(ev_h2, 1, -1, ret_jacobian=False)
    ax_perturbed.scatter(ev_h1[0], ev_h1[1], marker="s", edgecolors="black", color = "blue", linewidths=1, label=f"h1 {i}")
    ax_perturbed.scatter(ev_h2[0], ev_h2[1], marker="o", edgecolors="black", color = "orange", linewidths=1, label=f"h2 {i}")

# ax_perturbed.legend(loc="lower left")
plt.show()

breakpoint()