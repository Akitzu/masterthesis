import numpy as np
from mayavi import mlab
import matplotlib.pyplot as plt
import pickle
import horus as ho

bs, bsh, (nfp, coils, ma, sc_fieldline) = ho.ncsx()

guess = [1.527, 0.]

from pyoculus.problems import SimsoptBfieldProblem
from pyoculus.solvers import FixedPoint

ps = SimsoptBfieldProblem(bs, R0=ma.gamma()[0,0], Z0=0., Nfp=3)

fp = FixedPoint(ps)

fp.find(7, guess, niter=100, nrestart=0, tol=1e-8)

breakpoint()