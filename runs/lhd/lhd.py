import numpy as np
from pyoculus.problems import SimsoptBfieldProblem
from simsopt.field import BiotSavart
from simsopt.field import load_coils_from_makegrid_file

coils = load_coils_from_makegrid_file('LHD_data/lhd.coils.txt', order=6)
mf = BiotSavart(coils)
ps = SimsoptBfieldProblem.without_axis([3.9, 0], 5, mf, interpolate=False)
RZ0 = [ps._R0, ps._Z0]

from simsopt.geo import CurvePlanarFourier
qpts_phi = np.linspace(0, 1, 50, endpoint=False)
dofs = np.concatenate(([ps._R0], np.zeros(7)))
curve = CurvePlanarFourier(qpts_phi, 0, 1, False)
curve.set_dofs(dofs)

from simsopt.geo import SurfaceRZFourier
s = SurfaceRZFourier.from_nphi_ntheta(
    mpol=5,
    ntor=5,
    stellsym=True,
    nfp=5,
    range="full torus",
    nphi=64,
    ntheta=24,
)
s.fit_to_curve(curve, 1.5, flip_theta=False)

pyoproblem = SimsoptBfieldProblem(ps._R0, ps._Z0, 5, mf, interpolate=True, surf=s, degree=3, n=60)

print('Finished setting up the problem')

phis = [0]    #[(i / 4) * (2 * np.pi / nfp) for i in range(4)]

nfieldlines = 2
p1 = np.array([pyoproblem._R0, pyoproblem._Z0])
p2 = np.array([pyoproblem._R0, pyoproblem._Z0 + 1])
Rs = np.linspace(p1[0], p2[0], nfieldlines)
Zs = np.linspace(p1[1], p2[1], nfieldlines)

RZs = np.array([[r, z] for r, z in zip(Rs.flatten(), Zs.flatten())])

from horus import poincare
pplane = poincare(pyoproblem._mf, RZs, phis, pyoproblem.surfclassifier, tol = 1e-10, plot=False)

pplane.save('pplane.pkl')

# print('Finished setting up the problem')

# from pyoculus.problems import FixedPoint
# # set up the integrator
# iparams = dict()
# iparams["rtol"] = 1e-13
# # set up the point finder
# pparams = dict()
# pparams["nrestart"] = 0
# pparams["tol"] = 1e-15
# pparams['niter'] = 100
# # pparams["Z"] = 0
# fp_x1 = FixedPoint(pyoproblem, pparams, integrator_params=iparams)
# fp_x1.compute(guess=[pyoproblem._R0, 0.385], pp=10, qq=5, sbegin=3, send=5, checkonly=True)
