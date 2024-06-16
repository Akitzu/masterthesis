from pyoculus.toybox import psi_circularcurrentloop, psi_squared
import matplotlib.pyplot as plt
plt.style.use("lateky")
import matplotlib.lines as mlines
import numpy as np

### Creating the meshgrid and check at the psi values
r = np.linspace(1, 14, 60)
z = np.linspace(-6, 2.5, 60)
R, Z = np.meshgrid(r, z)

separatrix = {"R": 6, "Z": -5.5}
equilibrium = {"R": 6, "Z": 0}

psi_b = np.array([psi_squared([rr, 0., zz], **equilibrium) for rr, zz in zip(R.flatten(), Z.flatten())]).reshape(R.shape)
psi_sep = np.array([-10*psi_circularcurrentloop([rr, 0., zz], **separatrix) for rr, zz in zip(R.flatten(), Z.flatten())]).reshape(R.shape)

### Plot the flux surfaces
fig, ax = plt.subplots()

ax.set_xlim(1, 14)
ax.set_ylim(-6, 2.5)

contour1 = ax.contour(R, Z, psi_b, levels=50, colors="navy", alpha=0.3)
contour2 = ax.contour(R, Z, psi_sep, levels=40, linewidths=1,
                      colors="orangered", alpha=1, linestyles="-")

# Create proxy artists
line1 = mlines.Line2D([], [], color='navy', alpha=1, label=r"$\psi_{eq}$")
line2 = mlines.Line2D([], [], color='orangered', alpha=1, label=r"$\psi_{sep}$")

# Create legend from proxy artists
ax.legend(handles=[line1, line2], framealpha=1, fontsize=12, loc="upper right")

ax.scatter(equilibrium["R"], equilibrium["Z"], marker="o", color="tab:blue", edgecolors="black", linewidths=1, zorder=10)
ax.scatter(separatrix["R"], separatrix["Z"], marker="o", color="tab:orange", edgecolors="black", linewidths=1, zorder=10)

ax.set_xlabel(r"R", fontsize=16)
ax.set_ylabel(r"Z", fontsize=16)
ax.set_aspect('equal')

# ratio = 16/9
# xlims = ax.get_xlim()
# ylims = ax.get_ylim()
# ax.set_xlim(xlims[0], xlims[0]+ratio*(ylims[1]-ylims[0]))

fig.savefig("figs/sepflux.pdf", dpi=300, bbox_inches="tight", pad_inches=0.1)
fig.savefig("figs/sepflux.png", dpi=300, bbox_inches="tight", pad_inches=0.1)