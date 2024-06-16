from pyoculus.toybox import psi_circularcurrentloop, psi_squared
import matplotlib.pyplot as plt
plt.style.use("lateky")
import matplotlib.lines as mlines
import numpy as np

### Creating the meshgrid and check at the psi values
r = np.linspace(3.5, 9.2, 40)
z = np.linspace(-6, 2.5, 40)
R, Z = np.meshgrid(r, z)

separatrix = {"R": 6, "Z": -5.5}
equilibrium = {"R": 6, "Z": 0}

psi_b = np.array([psi_squared([rr, 0., zz], **equilibrium) for rr, zz in zip(R.flatten(), Z.flatten())]).reshape(R.shape)
psi_sep = np.array([-10*psi_circularcurrentloop([rr, 0., zz], **separatrix) for rr, zz in zip(R.flatten(), Z.flatten())]).reshape(R.shape)

### Plot the flux surfaces
fig, ax = plt.subplots()

ax.set_xlim(3.5, 9.2)
ax.set_ylim(-6, 2.5)

contour1 = ax.contour(R, Z, psi_b, levels=50, colors="black", alpha=0.7)
contour2 = ax.contour(R, Z, psi_sep, levels=50, colors="red", alpha=0.9)

# Create proxy artists
line1 = mlines.Line2D([], [], color='black', alpha=0.5, label=r"$\psi_{eq}$")
line2 = mlines.Line2D([], [], color='red', alpha=0.5, label=r"$\psi_{sep}$")

# Create legend from proxy artists
ax.legend(handles=[line1, line2], framealpha=1)

ax.scatter(equilibrium["R"], equilibrium["Z"], marker="o", edgecolors="black", linewidths=1, zorder=10)
ax.scatter(separatrix["R"], separatrix["Z"], marker="o", edgecolors="black", linewidths=1, zorder=10)

ax.set_xlabel(r"R")
ax.set_ylabel(r"Z")
ax.set_aspect('equal')

fig.savefig("sepflux.pdf")
fig.savefig("sepflux.png")

plt.show()