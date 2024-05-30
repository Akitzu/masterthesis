from simsopt._core import load
# replace "NAME_OF_FILE_YOU_DOWNLOADED" with the name you gave the file
surfaces, ma, coils = load(f'serial0928241.json')

from pyoculus.problems import SimsoptBfieldProblem
from simsopt.geo import SurfaceRZFourier

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

R0, _, Z0 = ma.gamma()[0,:]
ps = SimsoptBfieldProblem.from_coils(R0=R0, Z0=Z0, Nfp=3, coils=coils, interpolate=True, surf=s) # ncoils=2, mpol=5, ntor=5, n=40, h=0.05)

from pyoculus.problems import FixedPoint
# set up the integrator
iparams = dict()
iparams["rtol"] = 1e-12

pparams = dict()
pparams["nrestart"] = 0
pparams["tol"] = 1e-15
pparams['niter'] = 100
# pparams["Z"] = 0 

fp11_1 = FixedPoint(ps, pparams, integrator_params=iparams)
fp11_1.compute(guess=[1.4446355574662593, 0.0], pp=3, qq=6, sbegin=0.4, send=1.6, checkonly=True)

fp11_2 = FixedPoint(ps, pparams, integrator_params=iparams)
fp11_2.compute(guess=[1.346295615988142, 0.2133036397909969], pp=3, qq=6, sbegin=0.4, send=1.6, checkonly=True)

from pyoculus.solvers import Manifold
iparam = dict()
iparam["rtol"] = 1e-13

mp = Manifold(ps, fp11_2, fp11_1, integrator_params=iparam)

mp.choose(signs=[[-1, -1],[-1, -1]])

mp.compute(nintersect = 6, epsilon=1e-6, neps = 1, directions="inner")