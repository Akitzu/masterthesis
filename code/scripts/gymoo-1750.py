from simsopt.configs import get_w7x_data
from simsopt.field import Current
from mpi4py import MPI
import numpy as np
import datetime
import pickle
import argparse

# Adding the path to the horus package
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import horus as ho


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convergence domain for GYM000-1750")


    options = {
        "pp": 3,
        "qq": 7,
        "sbegin": 1.2,
        "send": 1.,
        "tol": 1e-8,
        "rtol": 1e-10,
        "checkonly": True,
        "eps": 1e-5,
    }

    # Get the W7-X data for the GYM000-1750 configuration
    w7x = get_w7x_data()

    currents = [Current(1.109484) * 1e6 for _ in range(5)]
    currents.append(Current(-0.3661) * 1e6)
    currents.append(Current(-0.3661) * 1e6)

    bs, bsh, (nfp, coils, ma, sc_fieldline) = ho.stellarator(w7x[0], currents, w7x[2], nfp=5, surface_radius=2)

    # Compute the convergence domain
    ps = ho.SimsoptBfieldProblem(ma.gamma()[0, 0], 0, 5, bs)
    
    R = np.linspace(1.2, 1.8, 31)
    Z = np.linspace(-0.6, 0.6, 61)

    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()  # Rank of the current process
    size = comm.Get_size()  # Total number of processes
    
    # Divide the work among the processes
    R_split = np.array_split(R, size)[rank]
    Z_split = np.array_split(Z, size)[rank]

    # Perform the calculation for this process's subset of R and Z
    convdom_checkonly_local = ho.convergence_domain(ps, R_split, Z_split)

    # Gather the results to the root process
    convdom_checkonly = comm.gather(convdom_checkonly_local, root=0)

    # The root process saves the result
    if rank == 0:
        # Flatten the list of results
        convdom_checkonly = np.concatenate(convdom_checkonly)

        # Save the result
        date = datetime.datetime.now().strftime("%Y%m%d")
        dumpname = f"convergence_domain_GYM000-1750_{date}.pkl"
        with open(os.path.join('..', 'output', dumpname), 'wb') as f:
            pickle.dump(convdom_checkonly, f)