from mpi4py import MPI
import numpy as np
from area import homoclinics
import pickle

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Define the parameters to be run
amp = np.linspace(1e-10, 1, 288)
amp = np.array_split(amp, comm.Get_size())

amp = amp[rank]

# Run the function
results = []
for a in amp:
    m, n = 18, -3
    print(f"{rank} - Running for m = {m}, n = {n}, amplitude = {a:.5f}")
    try:
        result = homoclinics(m, n, a)
        print(f"{rank} - finished")
        results.append([a, result])
    except Exception as e:
        print(f"{rank} - failed")
        print(e)


print(f"{rank} - gathering and saving")
# Gather results from all processes to the root process
results = comm.gather(results, root=0)

# If this is the root process, save the results
if rank == 0:
    print("Saving")
    with open('results.pkl', 'wb') as f:
        pickle.dump(results, f)
