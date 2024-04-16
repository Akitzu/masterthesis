from mpi4py import MPI
import numpy as np
import subprocess

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

amplitudes = np.linspace(1e-3, 10, 10)
amplitudes = np.array_split(amplitudes, comm.Get_size())
amplitudes = amplitudes[rank]

for amp in amplitudes:
    print("rank: {}, amp: {}".format(rank, amp))
    subprocess.run(["python", "toybox_manifold.py", "-c", "-a", str(amp), "-n", "manifold_a"+str(amp)])
