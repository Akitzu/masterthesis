from simsopt.configs import get_w7x_data
from simsopt.field import Current
from simsopt.util import comm_world, proc0_print
import numpy as np
import datetime
import pickle
import argparse

# Adding the path to the horus package
import sys
import os
ROOT_DIR = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(os.path.abspath(ROOT_DIR))
import horus as ho


if __name__ == '__main__':
    args = argparse.ArgumentParser(description="Poincare for GYM000-1750")
    args.add_argument("-phis", nargs='+', type=float, default=[0], help="Phi coordinates for the Poincare plot")
    args.add_argument("-tol", type=float, default=1e-10, help="Tolerance for the Poincare plot")
    # parser.add_argument("-r", nargs='+', type=float, help="R coordinates for the Poincare plot")
    # parser.add_argument("-z", nargs='+', type=float, help="Z coordinates for the Poincare plot") 

    # Setup the lines initial coordinates    
    nfieldlines = 30
    Rs = np.linspace(6.05, 6.2, nfieldlines)
    Zs = [ma.gamma()[0, 2] for _ in range(nfieldlines)]
    RZs = np.array([[r, z] for r, z in zip(Rs, Zs)])

    nfieldlines = 5
    p1 = np.array([5.6144507858315915, -0.8067790944375764])
    p2 = np.array([5.78, -0.6])
    Rs = np.linspace(p1[0], p2[0], nfieldlines)
    Zs = np.linspace(p1[1], p2[1], nfieldlines)
    Rs, Zs = np.meshgrid(Rs, Zs)
    RZs2 = np.array([[r, z] for r, z in zip(Rs.flatten(), Zs.flatten())])

    RZs = np.concatenate((RZs, RZs2))   

    # Current configuration
    currents = [Current(1.109484) * 1e6 for _ in range(5)]
    currents.append(Current(-0.3661) * 1e6)
    currents.append(Current(-0.3661) * 1e6)

    # Get the W7-X data for the GYM000-1750 configuration
    w7x = get_w7x_data()

    bs, bsh, (nfp, coils, ma, sc_fieldline) = ho.stellarator(w7x[0], currents, w7x[2], nfp=5, surface_radius=2)

    proc0_print("Configuration loaded. Computing Poincare plot...")

    # Compute the Poincare plot
    _, _, fig, ax = ho.poincare(bsh, RZs, args.phis, sc_fieldline, tol = args.tol, comm = comm_world)

    proc0_print("Poincare plot computed. Saving...")

    # Save the plot
    if comm_world is None or comm_world.rank == 0:
        for col in ax[0,0].collections:
            col.set_color('black')
            col.set_sizes([0.5])

        date = datetime.datetime.now().strftime("%d%m%Y_%H%M")
        dumpname = f"poincare_GYM000-1750_{date}.pkl"
        with open(os.path.join(ROOT_DIR, "output/", dumpname), 'wb') as f:
                    pickle.dump(fig, f)

    proc0_print("Poincare plot saved.")