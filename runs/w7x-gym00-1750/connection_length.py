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
from horus import plot_poincare_simsopt
from plotting_script import make_plot_phi0

ratio = 16/9
DPI = 600
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
        
    
def plot_manifold(fieldlines_phi_hits, ax, stable = True, ii = 0, stop = None, end = -1, **kwargs):
    options = {
        "color": "black",
        "linewidth": 1,
        "zorder": 10,
        "marker": ".",
        "markersize": 1,
    }
    options.update(kwargs)

    arr_phi_hits = np.array([[i, *phi] for i, phis in enumerate(fieldlines_phi_hits) for phi in phis])
    sorted_arr = arr_phi_hits[np.argsort(arr_phi_hits[:, 1])]
    
    if stable:
        indixes = (ii + np.arange(5)) % 5
    else:
        indixes = (ii + np.arange(5) * 4) % 5

    i, id, iter = 0, 0, 0
    rzs = np.array([[], []]).T
    while i < sorted_arr.shape[0]:
        i_s, i_e = extract(sorted_arr, i, indixes[id])
        print(f"{iter} - Information {i}, {i_s}, {i_e}, {indixes[id]}")
        group = sorted_arr[i_s:i_e].copy()        
        group = group[group[:, 0].argsort()]

        rr = np.sqrt(group[:, 3] ** 2 + group[:, 4] ** 2)
        z = group[:, 5]
        rzs = np.vstack((rzs, np.array([rr, z]).T))

        if i_e == -1:
            break
        else:
            id = (np.where(indixes == sorted_arr[i_s,2])[0][0] + 1) % 5
            i = i_e
            iter += 1
        if stop is not None and iter > stop:
            break
        
    if end != -1:
        rzs = rzs[:end]
    print(f"Number of points is {rzs.shape[0]}")

    ax.plot(*rzs.T, **options)
    ax.set_xlabel(r"R [m]")
    ax.set_ylabel(r"Z [m]")
    ax.set_aspect("equal")
    return ax.get_figure(), ax


fig, ax = plt.subplots()
ax.grid(False)

# make_plot_phi0(ax, lcdata_phi0_filename, tpl=(0.7, 0.1, 0.03, 0.8))

tys, phi_hits = pickle.load(open(file_poincare, "rb"))
plot_poincare_simsopt(phi_hits, ax)

# # CU
# ax.set_xlim(5.3, 5.9)
# ax.set_ylim(0.5, 1.1)

# tys, phi_hits = pickle.load(open(file_manifold_stable, "rb"))
# plot_manifold(phi_hits, ax, ii=0, stop = 11, color="navy")
# fig.savefig(saving_folder / "gym00_1750_connlength_1.png", dpi = DPI, bbox_inches='tight', pad_inches=0.1)

# tys, phi_hits = pickle.load(open(file_manifold_unstable, "rb"))
# plot_manifold(phi_hits, ax, stable=False, ii = 4, stop = 10, end=-20, color="orangered")
# fig.savefig(saving_folder / "gym00_1750_connlength_2.png", dpi = DPI, bbox_inches='tight', pad_inches=0.1)

# tys, phi_hits = pickle.load(open(file_manifold_stable, "rb"))
# plot_manifold(phi_hits, ax, ii=2, stop = 8, color="black")
# fig.savefig(saving_folder / "gym00_1750_connlength_3.png", dpi = DPI, bbox_inches='tight', pad_inches=0.1)

# tys, phi_hits = pickle.load(open(file_manifold_unstable, "rb"))
# plot_manifold(phi_hits, ax, stable=False, ii = 3, stop = 7, end=-50, color="maroon")
# fig.savefig(saving_folder / "gym00_1750_connlength_4.png", dpi = DPI, bbox_inches='tight', pad_inches=0.1)

# General
ax.set_xlim(5.3, 6.2)
ax.set_ylim(-1.2, 1.2)

tys, phi_hits = pickle.load(open(file_manifold_stable, "rb"))
plot_manifold(phi_hits, ax, ii=0, stop = 9, end=-30, color="green")
# plot_manifold(phi_hits, ax, ii=2, stop = 9, end=-200, color="black")
# plot_manifold(phi_hits, ax, ii=3, stop = 9, end=-160, color="black")
# plot_manifold(phi_hits, ax, ii=4, stop = 9, end=-157, color="black")
# fig.savefig(saving_folder / "gym00_1750_connlength_ncu_1.png", dpi = DPI, bbox_inches='tight', pad_inches=0.1)

tys, phi_hits = pickle.load(open(file_manifold_unstable, "rb"))
# plot_manifold(phi_hits, ax, stable=False, ii = 0, stop = 7, end=-35, color="red")
# plot_manifold(phi_hits, ax, stable=False, ii = 2, stop = 7, end=-20, color="red")
# plot_manifold(phi_hits, ax, stable=False, ii = 3, stop = 7, end=-50, color="red")
plot_manifold(phi_hits, ax, stable=False, ii = 4, stop = 8, end=-160, color="red")
# fig.savefig(saving_folder / "gym00_1750_connlength_ncu_2.png", dpi = DPI, bbox_inches='tight', pad_inches=0.1)



# fig.savefig(saving_folder / "gym00_1750_connlength_ncu.png", dpi = DPI, bbox_inches='tight', pad_inches=0.1)

# tys, phi_hits = pickle.load(open(file_manifold_stable, "rb"))
# plot_manifold(phi_hits, ax, ii=2, stop = 8, color="black")
# fig.savefig(saving_folder / "gym00_1750_connlength_ncu_3.png", dpi = DPI, bbox_inches='tight', pad_inches=0.1)

# tys, phi_hits = pickle.load(open(file_manifold_unstable, "rb"))
# plot_manifold(phi_hits, ax, stable=False, ii = 3, stop = 7, end=-50, color="maroon")
# fig.savefig(saving_folder / "gym00_1750_connlength_ncu_4.png", dpi = DPI, bbox_inches='tight', pad_inches=0.1)

fig.show()