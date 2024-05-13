from mpi4py import MPI
import numpy as np
from homoclinics import homoclinics

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Define the parameters to be run
amp = np.linspace(0., 0.1, 100)
amp = np.array_split(amp, comm.Get_size())

amp = amp[rank]

# Run the function
results = []
for a in amp:
    m, n = 18, -3
    print(f"{rank} - Running for m = {m}, n = {n}, amplitude = {a:.5f}")
    try:
        result = homoclinics(m, n, a)
        results.append([a, *result])
    except ValueError as e:
        print(e)

# Convert results to a numpy array
results = np.array(results)

# Gather results from all processes to the root process
results = comm.gather(results, root=0)

# If this is the root process, print the results
if rank == 0:
    np.save('results.npy', results)