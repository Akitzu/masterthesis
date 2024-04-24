from mpi4py import MPI
import numpy as np
from pythonrun import fun

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

amplitudes = np.linspace(1e-3, 1, 2)
amplitudes = np.array_split(amplitudes, comm.Get_size())
amplitudes = amplitudes[rank]

for amp in amplitudes:
    print("rank: {}, amp: {}".format(rank, amp))
    try:
        fun(amp, fname_sigdigit = 3)
    except ValueError as e:
        print(e)
