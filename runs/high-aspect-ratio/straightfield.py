from pyoculus.problems import AnalyticCylindricalBfield
import matplotlib.pyplot as plt
plt.style.use('lateky')
import numpy as np
import pickle

# Creating the pyoculus problem object, adding the perturbation here use the R, Z provided as center point
pyoproblem = AnalyticCylindricalBfield(
    6,
    0,
    0.8875,
    0.2
)

rho = 1.75
phis = np.linspace(0, 2*np.pi, 20)
thetas = np.linspace(0, 2*np.pi, 100)

X, Y = np.meshgrid(thetas, phis)
# Flattening of the coordinates
iota = 2/3
# X = Y*iota

coords = np.array([[6+rho*np.cos(theta), phi, rho*np.sin(theta)] for theta, phi in zip(X.flatten(), Y.flatten())])
Bs = pyoproblem.B_many(*coords.T)
coords = coords.reshape(X.shape + (3,))
Bs = Bs.reshape(X.shape + (3,))
Bphi = Bs[:,:,1]
Btheta = (np.cos(X) * Bs[:,:,2] - np.sin(X) * Bs[:,:,0])/rho

# Normalization
norm = np.sqrt(Bphi**2 + Btheta**2)
Bphi /= norm
Btheta /= norm

fig, ax = plt.subplots()
ax.set_xlabel(r"$\phi$", fontsize=14)
ax.set_ylabel(r"$\theta$", fontsize=14)

# ax.quiver(Y, X, Bphi, Btheta, scale=4, scale_units='xy')
contour = ax.pcolor(Y, X, np.arctan(Btheta/Bphi), shading='auto', cmap='RdBu_r')
cbar = fig.colorbar(contour, ax=ax)
cbar.set_label(r"$\arctan(B^\theta/B^\phi)$", fontsize=12)

ax.grid(False)

fig.savefig(f"figs/fieldlines.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
fig.savefig(f"figs/fieldlines.pdf", dpi=300, bbox_inches='tight', pad_inches=0.1)