import matplotlib.pyplot as plt
plt.style.use('lateky')
import numpy as np
from horus import plot_poincare_pyoculus

file = "unperturbed"
# file = "perturbed_3_2"
xydata = np.load(f"data/{file}.npy")

fig, ax = plt.subplots()
plot_poincare_pyoculus(xydata, ax, xlims=None, ylims=None, linewidths=0.3)

ax.set_xlim(3.8, 8.2)
ax.set_ylim(-2.2, 2.2)

fig.savefig(f"figs/{file}.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
fig.savefig(f"figs/{file}.pdf", dpi=300, bbox_inches='tight', pad_inches=0.1)