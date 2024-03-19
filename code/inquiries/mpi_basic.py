from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Each process has an array of different length
data = [np.array([rank] * np.random.randint(2, 5)) for _ in range(rank+1)]


# Flatten the data and get the sizes
if len(data) != 0:
    flat_data = np.concatenate(data)
else:
    flat_data = np.array([], dtype='i')
sizes = [len(arr) for arr in data]


print("rank: {}, local_data: {}".format(rank, data))
print("rank: {}, local_flat_data: {}".format(rank, flat_data))
print("rank: {}, sizes: {}".format(rank, sizes))

# Gather the sizes on each process
all_sizes = comm.gather(sizes)

print("rank: {}, all_sizes: {}".format(rank, all_sizes))

if rank == 0:
    recvbuf = np.empty(sum(sum(all_sizes, [])), dtype='i')
    # Calculate displacements for the receive buffer
    count = [sum(sublist) for sublist in all_sizes]
    displacements = [0] + list(np.cumsum(count)[:-1])
    
    print("rank: {}, displacements: {}".format(rank, displacements))
    print("rank: {}, counts: {}".format(rank, count))
    print("rank: {}, recvbuf: {}".format(rank, recvbuf))
    
    recvbuf_args = (recvbuf, count, displacements, MPI.INT)
else:
    recvbuf_args = None

# Gather the data
comm.Gatherv(sendbuf = flat_data, recvbuf = recvbuf_args, root = 0)

if rank == 0:
    print("\n")
    print(recvbuf)
    print(type(recvbuf))
    data_unflattened = [np.split(arr, np.cumsum(sub_sizes)[:-1]) for arr, sub_sizes in zip(np.split(recvbuf, np.cumsum(count)[:-1]), all_sizes)]
    print("Unflattened data: ", data_unflattened)