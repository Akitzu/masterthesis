#!/usr/bin/env python

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys

latexplot_folder = Path("../../latex/images/plots").absolute()
saving_folder = Path("figs").absolute()

sys.path.append(str(latexplot_folder))
sys.path.append("connection_length")
from plot_poincare import plot_poincare_simsopt
from plotting_script import make_plot_phi0

ratio = 16/9
DPI = 300
file_poincare = "pkl/gym00_1750.pkl"
file_manifold_stable = "pkl/manifold_stable.pkl"
file_manifold_unstable = "pkl/manifold_unstable.pkl"
lcdata_phi0_filename = "connection_length/LC_phi0_all_targets.txt"

###############################################################################
# Plot and save the poincare plot
###############################################################################

fig, ax = plt.subplots()
ax.grid(False)

# make_plot_phi0(ax, lcdata_phi0_filename, tpl=(0.7, 0.1, 0.03, 0.8))
# # tys, phi_hits = pickle.load(open(file_poincare, "rb"))
# # plot_poincare_simsopt(phi_hits, ax)
# tys, phi_hits = pickle.load(open(file_manifold_stable, "rb"))
# plot_poincare_simsopt(phi_hits, ax, color="orangered")
# tys, phi_hits = pickle.load(open(file_manifold_unstable, "rb"))
# plot_poincare_simsopt(phi_hits, ax, color="red")

def extract(arr, ci, nidx):
    id_start = -1
    id_end = -1

    flag = False
    for i, row in enumerate(arr[ci:]):
        if row[2] == nidx and not flag:
            id_start = ci + i
            flag = True
        elif row[2] != nidx and flag:
            id_end = ci + i
            break

    return id_start, id_end
        
    
def plot_manifold(fieldlines_phi_hits, ax, ii = 0, end = None, **kwargs):
    options = {
        "color": "black",
        "linewidth": 1,
        "zorder": 10,
        "marker": ".",
        "markersize": 1,
    }
    options.update(kwargs)

    arr_phi_hits = np.array([[i, *phi] for i, phis in enumerate(fieldlines_phi_hits) for phi in phis])
    sorted_arr = arr_phi_hits[np.lexsort((arr_phi_hits[:, 2], arr_phi_hits[:, 0], arr_phi_hits[:, 1]))]
    
    indixes = (ii + np.arange(5) * 4) % 5
    
    i = 0
    while i < sorted_arr.shape[0]:
        i_s, i_e = extract(sorted_arr, i, indixes[i % 5])
        group = sorted_arr[i_s:i_e].copy()
        group = group[group[:, 0].argsort()]
        r = np.sqrt(group[:, 3] ** 2 + group[:, 4] ** 2)
        z = group[:, 5]

        ax.plot(r, z, **options)

        if i_e == -1:
            break
        else:
            i = i_e
        if end is not None and i > end:
            break

    ax.set_xlabel(r"R [m]")
    ax.set_ylabel(r"Z [m]")
    ax.set_aspect("equal")
    return ax.get_figure(), ax


fig, ax = plt.subplots()
ax.grid(False)

make_plot_phi0(ax, lcdata_phi0_filename, tpl=(0.7, 0.1, 0.03, 0.8))

tys, phi_hits = pickle.load(open(file_poincare, "rb"))
plot_poincare_simsopt(phi_hits, ax)

# tys, phi_hits = pickle.load(open(file_manifold_stable, "rb"))
# plot_poincare_simsopt(phi_hits, ax, color="orangered", linewidths=0.6, s=4)
# tys, phi_hits = pickle.load(open(file_manifold_unstable, "rb"))
# plot_poincare_simsopt(phi_hits, ax, color="black", linewidths=0.6, s=4)

# tys, phi_hits = pickle.load(open(file_manifold_stable, "rb"))
# plot_manifold(phi_hits, ax, end = 10000, color="orangered")
# tys, phi_hits = pickle.load(open(file_manifold_unstable, "rb"))
# plot_manifold(phi_hits, ax, ii = 3, end = 9000, color="black")
fig.show()


# ax.set_xlim(5.3, 6.2)
# ax.set_ylim(-1.2, 1.2)

# fig.savefig(saving_folder / "gym00_1750_connlength.png", dpi = DPI, bbox_inches='tight', pad_inches=0.1)