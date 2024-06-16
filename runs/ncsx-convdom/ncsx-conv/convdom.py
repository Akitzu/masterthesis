import numpy as np
import matplotlib.pyplot as plt
from pyoculus.problems import SimsoptBfieldProblem
import pickle
import horus as ho

bs, bsh, (nfp, coils, ma, sc_fieldline) = ho.ncsx()

ps = SimsoptBfieldProblem.from_coils(R0=ma.gamma()[0,0], Z0=0., Nfp=3, coils=coils, interpolate=True, ncoils=3)

R = np.linspace(1.2, 1.8, 1)
Z = np.linspace(-0.6, 0.6, 2)

convdom_checkonly = ho.convergence_domain(ps, R, Z, rtol = 1e-7, tol = 1e-8, eps = 1e-5, checkonly = True)

class TMPClass():
    def __init__(self):
        pass

fixed_points = list()
for fp in convdom_checkonly[1]:
    fp_tmp = TMPClass()
    fp_tmp.x = fp.x
    fp_tmp.y = fp.y
    fp_tmp.z = fp.z
    fp_tmp.successful = fp.successful
    fp_tmp.GreenesResidue = fp.GreenesResidue
    fixed_points.append(fp_tmp)
fixed_points = np.array(fixed_points, dtype=object)

with open("convdom_arr.npy", "wb") as f:
    np.save(f, convdom_checkonly[0][:-1].astype(float))

with open("convdom_fps.npy", "wb") as f:
    np.save(f, fixed_points)
