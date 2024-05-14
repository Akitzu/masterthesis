from mpi4py import MPI
import numpy as np
from homoclinics import homoclinics
import pickle

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Define the parameters to be run
amp = np.linspace(0., 1, 72)
amp = np.array_split(amp, comm.Get_size())

amp = amp[rank]

# Run the function
results = []
for a in amp:
    m, n = 6, -1
    print(f"{rank} - Running for m = {m}, n = {n}, amplitude = {a:.5f}")
    try:
        result = homoclinics(m, n, a)
        # result = [1, 2, 3, 4]
        results.append([a, *result[0]])
    except ValueError as e:
        print(e)

# Gather results from all processes to the root process
results = comm.gather(results, root=0)

# If this is the root process, save the results
if rank == 0:
    with open('results.pkl', 'wb') as f:
        pickle.dump(results, f)
