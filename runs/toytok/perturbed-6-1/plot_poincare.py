import matplotlib.pyplot as plt
plt.style.use('lateky')
import numpy as np
from horus import plot_poincare_pyoculus

# file = "poincare_cleaner"
# file = "perturbed_6_1"
file = "perturbed_6_1_4e-1"

xydata = np.load(f"data/{file}.npy")

fig, ax = plt.subplots()
plot_poincare_pyoculus(xydata, ax, linewidths=0.2)
fig.savefig(f"figs/{file}.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
fig.savefig(f"figs/{file}.pdf", dpi=300, bbox_inches='tight', pad_inches=0.1)

# # closer look
# fig, ax = plt.subplots()
# plot_poincare_pyoculus(xydata, ax, color=None, linewidths=0.5)
# ax.set_aspect("equal")
# ax.set_xlim(5, 7.5)
# ax.set_ylim(-6, -3.5)
# ax.scatter(
#     6.41409781, -0.69367863, marker="o", edgecolors="black", linewidths=1, zorder=11
#     )
# ax.scatter(
#     6.20365733, -4.49774606, marker="X", edgecolors="black", linewidths=1, zorder=11
# )

# fig.savefig(f"figs/{file}_closer.pdf", dpi=300, bbox_inches='tight', pad_inches=0.1)
# fig.savefig(f"figs/{file}_closer.png", dpi=300, bbox_inches='tight', pad_inches=0.1)