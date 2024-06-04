from pyoculus.problems import SimsoptBfieldProblem
from pyoculus.solvers import FixedPoint, Manifold
from simsopt.geo import SurfaceRZFourier
import matplotlib.pyplot as plt
from simsopt._core import load
from horus import poincare
import pandas as pd
import numpy as np

surfaces, ma, coils = load(f'serial0928241.json')

s = SurfaceRZFourier.from_nphi_ntheta(
    mpol=5,
    ntor=5,
    stellsym=True,
    nfp=3,
    range="full torus",
    nphi=64,
    ntheta=24,
)
s.fit_to_curve(ma, 0.7, flip_theta=False)

# Setting the problem
R0, _, Z0 = ma.gamma()[0,:]
ps = SimsoptBfieldProblem.from_coils(R0=R0, Z0=Z0, Nfp=3, coils=coils, interpolate=True, surf=s)

# Poincare plot
phis = [0]    #[(i / 4) * (2 * np.pi / nfp) for i in range(4)]

nfieldlines = 10
Rs = np.linspace(0.884, 1.2, nfieldlines)
Zs = np.zeros_like(Rs)
RZs = np.array([[r, z] for r, z in zip(Rs, Zs)])

nfieldlines = 60
p1 = np.array([1.09955, 0.0712])
p2 = np.array([1.4016, 0.1072])
Rs = np.linspace(p1[0], p2[0], nfieldlines)
Zs = np.linspace(p1[1], p2[1], nfieldlines)
# Rs, Zs = np.meshgrid(Rs, Zs)
RZs2 = np.array([[r, z] for r, z in zip(Rs.flatten(), Zs.flatten())])
RZs = np.concatenate((RZs, RZs2))

nfieldlines = 10
p1 = np.array([1.385, 0.])
p2 = np.array([1.526, 0.])
Rs = np.linspace(p1[0], p2[0], nfieldlines)
Zs = np.linspace(p1[1], p2[1], nfieldlines)
RZs2 = np.array([[r, z] for r, z in zip(Rs.flatten(), Zs.flatten())])
RZs = np.concatenate((RZs, RZs2))

nfieldlines = 10
p1 = np.array([1.4446, 0.])
p2 = np.array([1.4822, 0.])
Rs = np.linspace(p1[0], p2[0], nfieldlines)
Zs = np.linspace(p1[1], p2[1], nfieldlines)
RZs2 = np.array([[r, z] for r, z in zip(Rs.flatten(), Zs.flatten())])
RZs = np.concatenate((RZs, RZs2))

pplane = poincare(ps._mf_B, RZs, phis, ps.surfclassifier, tmax = 15000, tol = 1e-10, plot=False)
fig, ax = pplane.plot(phis)
ax = ax[0,0]

# Finding all fixedpoints

# set up the integrator
iparams = dict()
iparams["rtol"] = 1e-12

pparams = dict()
pparams["nrestart"] = 0
pparams["tol"] = 1e-15
pparams['niter'] = 100
# pparams["Z"] = 0 

fp11_o1 = FixedPoint(ps, pparams, integrator_params=iparams)
fp11_o1.compute(guess=[1.4446355574662593, 0.0], pp=3, qq=6, sbegin=0.4, send=1.6, checkonly=True)
fp11_o2 = FixedPoint(ps, pparams, integrator_params=iparams)
fp11_o2.compute(guess=[1.346295615988142, 0.2133036397909969], pp=3, qq=6, sbegin=0.4, send=1.6, checkonly=True)
fp11_o3 = FixedPoint(ps, pparams, integrator_params=iparams)
fp11_o3.compute(guess=[1.40150403, 0.10815878], pp=3, qq=6, sbegin=0.4, send=1.6, checkonly=True)
fp11_o4 = FixedPoint(ps, pparams, integrator_params=iparams)
fp11_o4.compute(guess=[1.40150403, -0.10815878], pp=3, qq=6, sbegin=0.4, send=1.6, checkonly=True)
fp11_x1 = FixedPoint(ps, pparams, integrator_params=iparams)
fp11_x1.compute(guess=[1.43378117, 0.05140443], pp=3, qq=6, sbegin=0.4, send=1.6, checkonly=True)
fp11_x2 = FixedPoint(ps, pparams, integrator_params=iparams)
fp11_x2.compute(guess=[1.43378117, -0.05140443], pp=3, qq=6, sbegin=0.4, send=1.6, checkonly=True)

for fp in [fp11_x1, fp11_x2]:
    results11 = [list(p) for p in zip(fp.x, fp.y, fp.z)]
    for rr in results11:
        ax.scatter(rr[0], rr[2], marker="X", edgecolors="black", linewidths=1)
for fp in [fp11_o1, fp11_o2, fp11_o3, fp11_o4]:
    results11 = [list(p) for p in zip(fp.x, fp.y, fp.z)]
    for rr in results11:
        ax.scatter(rr[0], rr[2], marker="o", edgecolors="black", linewidths=1)

data = [
    {'r': fp11_x1.x[0], 'z': fp11_x1.z[0], 'GreenesResidue': fp11_x1.GreenesResidue},
    {'r': fp11_x2.x[0], 'z': fp11_x2.z[0], 'GreenesResidue': fp11_x2.GreenesResidue},
    {'r': fp11_o1.x[0], 'z': fp11_o1.z[0], 'GreenesResidue': fp11_o1.GreenesResidue},
    {'r': fp11_o2.x[0], 'z': fp11_o2.z[0], 'GreenesResidue': fp11_o2.GreenesResidue},
    {'r': fp11_o3.x[0], 'z': fp11_o3.z[0], 'GreenesResidue': fp11_o3.GreenesResidue},
    {'r': fp11_o4.x[0], 'z': fp11_o4.z[0], 'GreenesResidue': fp11_o4.GreenesResidue},
]
df = pd.DataFrame(data)

# Working on manifold
iparam = dict()
iparam["rtol"] = 1e-13

mp = Manifold(ps, fp11_x2, fp11_x1, integrator_params=iparam)
mp.choose(signs=[[1, 1],[1, -1]])

mp.compute(nintersect = 4, epsilon=1e-6, neps = 20)
mp.plot(ax=ax)

# Inner manifold
print("Working on Inner manifold")
mp.onworking = mp.inner
mp.find_clinic_single(0.001276810579762792, 0.0012768113453997163, n_s=2, n_u=2)
mp.find_clinic_single(0.005129109370459298, 0.0051291087795083574, n_s=2, n_u = 1, tol=1e-8)
mp.turnstile_area()

marker = ["X", "o", "s", "p", "P", "*", "x", "D", "d", "^", "v", "<", ">"]
confns = mp.onworking["find_clinic_configuration"]
n_u = confns["n_u"]+confns["n_s"]+2

for i, clinic in enumerate(mp.onworking["clinics"]):
    eps_s_i, eps_u_i = clinic[1:3]
    
    hu_i = mp.integrate(mp.onworking["rfp_u"] + eps_u_i * mp.onworking["vector_u"], n_u, 1)
    ax.scatter(hu_i[0,:], hu_i[1,:], marker=marker[i], color="royalblue", edgecolor='cyan', zorder=10, label=f'$h_{i+1}$')

# Outer manifold
print("Working on Outer manifold")
mp.onworking = mp.outer
mp.find_clinic_single(0.0015488037705831256, 0.0015488037607238807, n_s=2, n_u=2)
mp.find_clinic_single(0.0006060200774938109, 0.0006060193763593331, n_s=3, n_u=2)
mp.turnstile_area()

for i, clinic in enumerate(mp.onworking["clinics"]):
    eps_s_i, eps_u_i = clinic[1:3]
    
    hu_i = mp.integrate(mp.onworking["rfp_u"] + eps_u_i * mp.onworking["vector_u"], n_u, 1)
    ax.scatter(hu_i[0,:], hu_i[1,:], marker=marker[i], color="royalblue", edgecolor='cyan', zorder=10, label=f'$h_{i+1}$')

# Alternating hyperbolic point

mp_2 = Manifold(ps, fp11_o2, integrator_params=iparam)
mp_2.choose(signs=[[1, 1], [1, -1]])
mp_2.compute(nintersect = 4, epsilon=1e-6, neps = 20, directions="inner")
# mp_2.plot(ax=ax)

# Convergence figure

fig_conv, ax_conv = plt.subplots()

# ar = np.zeros((2, 3))
for ii, pot in enumerate(mp.inner["potential_integrations"]):
    ns = min(len(pot[0]), len(pot[1]))
    # ar[ii,:] = pot[0][1:ns]-pot[1][:ns-1]
    ax_conv.scatter(1+np.arange(ns-1), pot[0][1:ns]-pot[1][:ns-1], zorder=10)

for ii, pot in enumerate(mp.outer["potential_integrations"]):
    ns = min(len(pot[0]), len(pot[1]))
    # ar[ii,:] = pot[0][1:ns]-pot[1][:ns-1]
    ax_conv.scatter(1+np.arange(ns-1), pot[0][1:ns]-pot[1][:ns-1], zorder=10)

ax_conv.set_xlabel('Iteration')
ax_conv.set_ylabel('Potential integration')