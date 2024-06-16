from pyoculus.problems import AnalyticCylindricalBfield
import matplotlib.pyplot as plt
plt.style.use('lateky')
import numpy as np
from horus import plot_iota_q

r = np.loadtxt("data/r-squared.txt")
q = np.loadtxt("data/q-squared.txt")
iota = np.loadtxt("data/iota-squared.txt")

fig, ax, axins = plot_iota_q(r, iota, q, r_shift = -6)

# ax.hlines(2/3, -0.1, 2.1, color='black', linestyle='--', linewidth=0.5)

fig.savefig("figs/iota_q.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
fig.savefig("figs/iota_q.pdf", dpi=300, bbox_inches='tight', pad_inches=0.1)
