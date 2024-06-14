#!/usr/bin/env python

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys

latexplot_folder = Path("../../latex/images/plots").absolute()
saving_folder = Path("figs").absolute()

sys.path.append(str(latexplot_folder))
from plot_poincare import plot_poincare_simsopt

ratio = 16/9
DPI = 300
file_poincare = "pkl/gym00_1750.pkl"
# file_poincare = "pkl/default.pkl"

###############################################################################
# Plot and save the poincare plot
###############################################################################

fig, ax = plt.subplots(dpi=DPI)
with open(file_poincare, "rb") as f:
    tys, phi_hits = pickle.load(open(file_poincare, "rb"))
    plot_poincare_simsopt(phi_hits, ax)

# ax.set_xlim(5.3, 6.3)
ax.set_xlim(5.1, 6.3)
ax.set_ylim(-1.2, 1.2)

plt.savefig(saving_folder / "gym00_1750_limits.png", bbox_inches='tight', pad_inches=0.1)
# plt.savefig(saving_folder / "default.png", bbox_inches='tight', pad_inches=0.1)