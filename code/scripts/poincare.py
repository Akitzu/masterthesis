from simsopt.configs import get_ncsx_data, get_w7x_data
from simsopt.field import Current
import numpy as np
import pickle
import argparse

# Adding the path to the horus package
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import horus as ho


def from_file(filename):
    with open(filename, "r") as f:
        args = f.readlines()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Poincare plot")
    parser.add_argument("-f", type=str, help="File from which to read settings")
    
    args = parser.parse_args()
    if args.f:
        args = from_file(args.f)   


    bs, bsh, (nfp, coils, ma, sc_fieldline) = ho.stellarator(w7x[0], currents, w7x[2], nfp=5, surface_radius=2)

    nfieldlines = 50
    phis = [0]    #[(i / 4) * (2 * np.pi / nfp) for i in range(4)]
    Rs = np.linspace(6.1, 6.2, nfieldlines)
    Zs = [ma.gamma()[0, 2] for _ in range(nfieldlines)]
    RZs = np.array([[r, z] for r, z in zip(Rs, Zs)])

    poincare = ho.poincare(bsh, RZs, phis, sc_fieldline, tol = 1e-10, engine="simsopt")

    pickle.dump(poincare, open("poincare.pkl", "wb"))