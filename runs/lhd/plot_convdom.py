from pathlib import Path
import matplotlib.pyplot as plt
plt.style.use('lateky')
from horus import plot_convergence_domain
import numpy as np
import pickle

class TMPClass():
    def __init__(self):
        pass

if __name__ == "__main__":
    arr = np.load("convdom_arr.npy")
    fps = np.load("convdom_fps.npy", allow_pickle=True)

    # fig = pickle.load(open("../../../../runs/unclassified-output/NCSX_poincare.pkl", "rb"))
    # ax = fig.get_axes()[0]
    fig, ax = plt.subplots()

    plot_convergence_domain(*arr, fps, ax)
    plt.show()
    fig.savefig("convdom.pdf")