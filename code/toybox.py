from pyoculus.problems import CylindricalBfield
from functools import partial
from jax import jit, jacfwd
import jax.numpy as jnp
import numpy as np


class AnalyticCylindricalBfield(CylindricalBfield):
    """Analytical Bfield problem class that allows adding analytical perturbations to an analytical equilibrium field. The equilibrium field is
    defined by the function `equ_squared(R, sf, shear)` and the perturbations can be choosen from the type dictionary. The possible types are:
        - "maxwell-boltzmann": Maxwell-Boltzmann distributed perturbation
        - "gaussian": Normally distributed perturbation

    Attributes:
        R (float): Major radius of the magnetic axis of the equilibrium field
        sf (float): Safety factor on the magnetic axis
        shear (float): Shear factor
        perturbations_args (list): List of dictionaries with the arguments of each perturbation
        amplitude (list): List of amplitudes of the perturbations. One can set the amplitude of all perturbations
            at once by setting this attribute:
            $ myBfield.amplitudes = [1, 2, 3]
            $ myBfield.amplitudes
            >> [1, 2, 3]
        perturbations (list): List of perturbations functions. To call a certain (for instance the first) perturbation one can do:
            $ myBfield.perturbations[0](rphiz)
            >> value

    Methods:
        set_amplitude(index, value): Set the amplitude of the perturbation at index to value
        set_perturbation(index, perturbation_args): Set the perturbation at index to be defined by perturbation_args
        add_perturbation(perturbation_args): Add a new perturbation defined by perturbation_args
        B_equilibrium(rphiz): Equilibrium field function
        dBdX_equilibrium(rphiz): Gradient of the equilibrium field function
        B_perturbation(rphiz): Perturbation field function
        dBdX_perturbation(rphiz): Gradient of the perturbation field function
    """

    def __init__(self, R, sf, shear, perturbations_args):
        """
        Args:
            R (float): Major radius of the magnetic axis of the equilibrium field
            sf (float): Safety factor on the magnetic axis
            shear (float): Shear factor
            perturbations_args (list): List of dictionaries with the arguments of each perturbation

        Example:
            $ pert1_dict = {m:2, n:-1, d:1, type: "maxwell-boltzmann", amplitude: 1e-2}
            $ pert2_dict = {m:1, n:0, d:1, type: "gaussian", amplitude: -1e-2}
            $ myBfield = AnalyticCylindricalBfield(R= 3, sf = 1.1, shear=3 pert=[pert1_dict, pert2_dict])
        """

        self.sf = sf
        self.shear = shear

        # Define the equilibrium field and its gradient
        self.B_equilibrium = jit(partial(equ_squared, R=R, sf=sf, shear=shear))
        self.dBdX_equilibrium = jit(lambda rr: jacfwd(self.B_equilibrium)(rr))

        # Define the perturbations and the gradient of the resulting field sum
        self._perturbations = [None] * len(perturbations_args)
        for pertdic in perturbations_args:
            pertdic.update({"R": R})

        self.perturbations_args = perturbations_args
        self._initialize_perturbations()

        # Call the CylindricalBfield constructor with (R,Z) of the axis
        super().__init__(R, 0)

    @property
    def amplitudes(self):
        return [pert["amplitude"] for pert in self.perturbations_args]

    @amplitudes.setter
    def amplitudes(self, value):
        self.amplitudes = value
        self._initialize_perturbations()

    def set_amplitude(self, index, value):
        """Set the amplitude of the perturbation at index to value"""
        self.amplitudes[index] = value
        self._initialize_perturbations(index)

    def set_perturbation(self, index, perturbation_args):
        self.perturbations_args[index] = perturbation_args
        self.perturbations_args[index].update({"R": self.R})
        self._initialize_perturbations(index)

    def add_perturbation(self, perturbation_args):
        self.perturbations_args.append(perturbation_args)
        self.perturbations_args[-1].update({"R": self.R})
        self._initialize_perturbations(len(self.perturbations_args) - 1)

    def _initialize_perturbations(self, index=None):
        if index is not None:
            indices = [index]
        else:
            indices = range(len(self.perturbations_args))

        for i in indices:
            tmp_args = self.perturbations_args[i].copy()
            tmp_args.pop("amplitude")
            tmp_args.pop("type")

            self._perturbations[i] = partial(
                PERT_TYPES_DICT[self.perturbations_args[i]["type"]], **tmp_args
            )

        self.B_perturbation = jit(
            lambda rr: jnp.sum(
                jnp.array(
                    [
                        pertdic["amplitude"] * self._perturbations[i](rr)
                        for i, pertdic in enumerate(self.perturbations_args)
                    ]
                ),
                axis=0,
            )
        )
        self.dBdX_perturbation = jit(lambda rr: jacfwd(self.B_perturbation)(rr))

    @property
    def perturbations(self):
        return [
            lambda rr: pertdic["amplitude"] * self._perturbations[i](rr)
            for i, pertdic in enumerate(self.perturbations_args)
        ]

    # BfieldProblem methods implementation
    def B(self, rr):
        B = self.B_equilibrium(rr) + self.B_perturbation(rr)
        return B.tolist()

    def dBdX(self, rr):
        dBdX = self.dBdX_equilibrium(rr) + self.dBdX_perturbation(rr)
        return dBdX.tolist()

    def B_many(self, r, phi, z, input1D=True):
        return jnp.array([self.B([r[i], phi[i], z[i]]) for i in range(len(r))]).tolist()

    def dBdX_many(self, r, phi, z, input1D=True):
        return jnp.array(
            [self.dBdX([r[i], phi[i], z[i]]) for i in range(len(r))]
        ).tolist()


## Equilibrium field


@jit
def equ_squared(rr, R, sf, shear):
    """
    Returns the B field derived from the psi and F flux functions given by
    """
    return jnp.array(
        [
            -2 * rr[2] / rr[0],
            (2 * sf + 2 * shear * (rr[2] ** 2 + (R - rr[0]) ** 2))
            * jnp.sqrt(R**2 - rr[2] ** 2 - (R - rr[0]) ** 2)
            / rr[0],
            (-2 * R + 2 * rr[0]) / rr[0],
        ]
    )


## Perturbation field


@jit
def pert_maxwellboltzmann(rr, R, d, m, n):
    return jnp.array(
        [
            jnp.sqrt(2)
            * (
                d**2
                * (
                    m
                    * (rr[2] ** 2 + (R - rr[0]) ** 2)
                    * jnp.imag(
                        (-R + rr[0] + 1j * rr[2]) ** (m - 1) * jnp.exp(1j * n * rr[1])
                    )
                    - 2
                    * rr[2]
                    * jnp.real((-R + rr[0] + 1j * rr[2]) ** m * jnp.exp(1j * n * rr[1]))
                )
                + rr[2]
                * (rr[2] ** 2 + (R - rr[0]) ** 2)
                * jnp.real((-R + rr[0] + 1j * rr[2]) ** m * jnp.exp(1j * n * rr[1]))
            )
            * jnp.exp(
                (-(R**2) + 2 * R * rr[0] - rr[0] ** 2 - rr[2] ** 2) / (2 * d**2)
            )
            / (jnp.sqrt(jnp.pi) * d**5 * rr[0]),
            0,
            jnp.sqrt(2)
            * (
                d**2
                * (
                    m
                    * (rr[2] ** 2 + (R - rr[0]) ** 2)
                    * jnp.real(
                        (-R + rr[0] + 1j * rr[2]) ** (m - 1) * jnp.exp(1j * n * rr[1])
                    )
                    - 2
                    * (R - rr[0])
                    * jnp.real((-R + rr[0] + 1j * rr[2]) ** m * jnp.exp(1j * n * rr[1]))
                )
                + (R - rr[0])
                * (rr[2] ** 2 + (R - rr[0]) ** 2)
                * jnp.real((-R + rr[0] + 1j * rr[2]) ** m * jnp.exp(1j * n * rr[1]))
            )
            * jnp.exp(
                (-(R**2) + 2 * R * rr[0] - rr[0] ** 2 - rr[2] ** 2) / (2 * d**2)
            )
            / (jnp.sqrt(jnp.pi) * d**5 * rr[0]),
        ]
    )


@jit
def pert_gaussian(rr, R, d, m, n):
    return jnp.array(
        [
            jnp.sqrt(2)
            * (
                d**2
                * m
                * jnp.imag(
                    (-R + rr[0] + 1j * rr[2]) ** (m - 1) * jnp.exp(1j * n * rr[1])
                )
                + rr[2]
                * jnp.real((-R + rr[0] + 1j * rr[2]) ** m * jnp.exp(1j * n * rr[1]))
            )
            * jnp.exp(
                (-(R**2) + 2 * R * rr[0] - rr[0] ** 2 - rr[2] ** 2) / (2 * d**2)
            )
            / (2 * jnp.sqrt(jnp.pi) * d**3 * rr[0]),
            0,
            jnp.sqrt(2)
            * (
                d**2
                * m
                * jnp.real(
                    (-R + rr[0] + 1j * rr[2]) ** (m - 1) * jnp.exp(1j * n * rr[1])
                )
                + (R - rr[0])
                * jnp.real((-R + rr[0] + 1j * rr[2]) ** m * jnp.exp(1j * n * rr[1]))
            )
            * jnp.exp(
                (-(R**2) + 2 * R * rr[0] - rr[0] ** 2 - rr[2] ** 2) / (2 * d**2)
            )
            / (2 * jnp.sqrt(jnp.pi) * d**3 * rr[0]),
        ]
    )


# Dictionary with the perturbation types
PERT_TYPES_DICT = {
    "maxwell-boltzmann": pert_maxwellboltzmann,
    "gaussian": pert_gaussian,
}


# Plot of the field intensity
def gaussian_psi(rr, R=3.0, d=0.1, m=2, n=1):
    return (
        ((rr[0] - R + rr[2] * 1j) ** m)
        * np.exp(-0.5 * ((-rr[0] + R) ** 2 + rr[2] ** 2) / d**2 + 1j * n * 0)
        / (np.sqrt(d**2) * np.sqrt(2 * np.pi))
    )


def mb_psi(rr, R=3.0, d=0.1, m=2, n=1):
    return (
        np.exp(-0.5 * ((-rr[0] + R) ** 2 + rr[2] ** 2) / d**2 + n * rr[1] * 1j)
        * np.sqrt(2 / np.pi)
        * (rr[0] - R + rr[2] * 1j) ** m
        * ((-rr[0] + R) ** 2 + rr[2] ** 2)
        / d**3
    )


def plot(ps, rw=[2, 5], zw=[-2, 2], nl=[100, 100]):
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))

    r = np.linspace(rw[0], rw[1], nl[0])
    z = np.linspace(zw[0], zw[1], nl[1])

    R, Z = np.meshgrid(r, z)
    Bs = np.array(
        [ps.B([r, 0.0, z]) for r, z in zip(R.flatten(), Z.flatten())]
    ).reshape(R.shape + (3,))
    mappable = axs[2].contourf(R, Z, np.linalg.norm(Bs, axis=2))
    fig.colorbar(mappable)

    Bs = np.array(
        [ps.B_equilibrium([r, 0.0, z]) for r, z in zip(R.flatten(), Z.flatten())]
    ).reshape(R.shape + (3,))
    mappable = axs[3].contourf(R, Z, np.linalg.norm(Bs, axis=2))
    fig.colorbar(mappable)

    R, Z = np.meshgrid(r, z)
    Bs = np.array(
        [ps.B_perturbation([r, 0.0, z]) for r, z in zip(R.flatten(), Z.flatten())]
    ).reshape(R.shape + (3,))
    mappable = axs[1].contourf(R, Z, np.linalg.norm(Bs, axis=2))
    fig.colorbar(mappable)

    psi = 0
    for pertdic in ps.perturbations_args:
        tmp_dict = pertdic.copy()
        tmp_dict.pop("amplitude")
        tmp_dict.pop("type")
        if pertdic["type"] == "maxwell-boltzmann":
            tmp_psi = np.array(
                [
                    mb_psi([r, 0.0, z], **tmp_dict)
                    for r, z in zip(R.flatten(), Z.flatten())
                ]
            ).reshape(R.shape)
        elif pertdic["type"] == "gaussian":
            tmp_psi = np.array(
                [
                    gaussian_psi([r, 0.0, z], **tmp_dict)
                    for r, z in zip(R.flatten(), Z.flatten())
                ]
            ).reshape(R.shape)
        psi += pertdic["amplitude"] * np.real(tmp_psi)

    mappable = axs[0].contourf(R, Z, psi)
    fig.colorbar(mappable)

    # Set the aspect equal
    for ax in axs:
        ax.set_aspect("equal")
        ax.scatter(ps._R0, 0, color="r", s=1)

    return fig, axs
