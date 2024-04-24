from mpi4py import MPI
import numpy as np
from homoclinics import homoclinics

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Define the parameters to be run
ms = np.arange(1, 20)
ms = np.array_split(ms, comm.Get_size())

ms = ms[rank]
ns = -1*np.ones_like(ms)

# Run the function
for m, n in zip(ms, ns):
    amplitude = np.power(10.,-(m-1))
    print(f"{rank} - Running for m = {m}, n = {n}, amplitude = {amplitude:.5f}")
    try:
        homoclinics(m, n, amplitude)
    except ValueError as e:
        print(e)
