from pyoculus.problems import CylindricalBfield
from functools import partial
from numba import njit
import numpy as np
import functools


def arraytize(f, shape="vector"):
    if shape == "vector":

        @functools.wraps(f)
        def wrapper(x):
            return np.array([f(xi) for xi in x])

    elif shape == "list":

        @functools.wraps(f)
        def wrapper(x):
            return np.array([f([x[0][i], x[1][i], x[2][i]]) for i in range(len(x[0]))])

    return wrapper


class AnalyticCylindricalBfield(CylindricalBfield):
    def __init__(
        self,
        B_equilibrium,
        B_perturbation,
        A_p=1e-3,
        equilibrium_args=dict(),
        perturbation_args=dict(),
    ):
        self.equilibrium_args = equilibrium_args
        self.perturbation_args = perturbation_args

        if "R" not in self.equilibrium_args.keys():
            raise ValueError(
                "A value for the equilibrium magnetic axis R must be in equilibrium_args"
            )

        super().__init__(self.equilibrium_args["R"], 0)
        self._A_p = A_p

        self.B_equilibrium = partial(B_equilibrium[0], **self.equilibrium_args)
        self.B_perturbation = partial(B_perturbation[0], **self.perturbation_args)
        self.dBdX_equilibrium = partial(B_equilibrium[1], **self.equilibrium_args)
        self.dBdX_perturbation = partial(B_perturbation[1], **self.perturbation_args)

    def B(self, rr):
        return self.B_equilibrium(rr) + self._A_p * self.B_perturbation(rr)

    def dBdX(self, rr):
        return self.dBdX_equilibrium(rr) + self._A_p * self.dBdX_perturbation(rr)

    def B_many(self, r, phi, z, input1D=True):
        return np.array([self.B([r[i], phi[i], z[i]]) for i in range(len(r))])

    def dBdX_many(self, r, phi, z, input1D=True):
        return np.array([self.dBdX([r[i], phi[i], z[i]]) for i in range(len(r))])


## Equilibrium field


@njit
def equ_squared(rr, R, sf, shear):
    """
    Returns the B field derived from the psi and F flux functions given by
    """
    return np.array(
        [
            -2 * rr[2] / rr[0],
            (2 * sf + 2 * shear * (rr[2] ** 2 + (R - rr[0]) ** 2))
            * np.sqrt(R**2 - rr[2] ** 2 - (R - rr[0]) ** 2)
            / rr[0],
            (-2 * R + 2 * rr[0]) / rr[0],
        ]
    )


@njit
def equ_squared_dBdX(rr, R, sf, shear):
    return np.array(
        [
            [2 * rr[2] / rr[0] ** 2, 0, -2 / rr[0]],
            [
                2
                * (
                    2
                    * rr[0]
                    * shear
                    * (R - rr[0])
                    * (-(R**2) + rr[2] ** 2 + (R - rr[0]) ** 2)
                    + rr[0]
                    * (R - rr[0])
                    * (sf + shear * (rr[2] ** 2 + (R - rr[0]) ** 2))
                    + (sf + shear * (rr[2] ** 2 + (R - rr[0]) ** 2))
                    * (-(R**2) + rr[2] ** 2 + (R - rr[0]) ** 2)
                )
                / (rr[0] ** 2 * np.sqrt(R**2 - rr[2] ** 2 - (R - rr[0]) ** 2)),
                0,
                2
                * rr[2]
                * (
                    -(R**2) * shear
                    + 6 * R * rr[0] * shear
                    - 3 * rr[0] ** 2 * shear
                    - 3 * rr[2] ** 2 * shear
                    - sf
                )
                / (rr[0] * np.sqrt(2 * R * rr[0] - rr[0] ** 2 - rr[2] ** 2)),
            ],
            [2 * R / rr[0] ** 2, 0, 0],
        ]
    )


## Perturbation field


@njit
def pert_maxwellboltzmann(rr, R, d, m, n):
    return np.array(
        [
            np.sqrt(2)
            * (
                d**2
                * (
                    m
                    * (rr[2] ** 2 + (R - rr[0]) ** 2)
                    * np.imag(
                        (-R + rr[0] + 1j * rr[2]) ** (m - 1) * np.exp(1j * n * rr[1])
                    )
                    - 2
                    * rr[2]
                    * np.real((-R + rr[0] + 1j * rr[2]) ** m * np.exp(1j * n * rr[1]))
                )
                + rr[2]
                * (rr[2] ** 2 + (R - rr[0]) ** 2)
                * np.real((-R + rr[0] + 1j * rr[2]) ** m * np.exp(1j * n * rr[1]))
            )
            * np.exp(
                (-(R**2) + 2 * R * rr[0] - rr[0] ** 2 - rr[2] ** 2) / (2 * d**2)
            )
            / (np.sqrt(np.pi) * d**5 * rr[0]),
            0,
            np.sqrt(2)
            * (
                d**2
                * (
                    m
                    * (rr[2] ** 2 + (R - rr[0]) ** 2)
                    * np.real(
                        (-R + rr[0] + 1j * rr[2]) ** (m - 1) * np.exp(1j * n * rr[1])
                    )
                    - 2
                    * (R - rr[0])
                    * np.real((-R + rr[0] + 1j * rr[2]) ** m * np.exp(1j * n * rr[1]))
                )
                + (R - rr[0])
                * (rr[2] ** 2 + (R - rr[0]) ** 2)
                * np.real((-R + rr[0] + 1j * rr[2]) ** m * np.exp(1j * n * rr[1]))
            )
            * np.exp(
                (-(R**2) + 2 * R * rr[0] - rr[0] ** 2 - rr[2] ** 2) / (2 * d**2)
            )
            / (np.sqrt(np.pi) * d**5 * rr[0]),
        ]
    )


@njit
def pert_gaussian(rr, R, d, m, n):
    return np.array(
        [
            np.sqrt(2)
            * (
                d**2
                * m
                * np.imag((-R + rr[0] + 1j * rr[2]) ** (m - 1) * np.exp(1j * n * rr[1]))
                + rr[2]
                * np.real((-R + rr[0] + 1j * rr[2]) ** m * np.exp(1j * n * rr[1]))
            )
            * np.exp(
                (-(R**2) + 2 * R * rr[0] - rr[0] ** 2 - rr[2] ** 2) / (2 * d**2)
            )
            / (2 * np.sqrt(np.pi) * d**3 * rr[0]),
            0,
            np.sqrt(2)
            * (
                d**2
                * m
                * np.real((-R + rr[0] + 1j * rr[2]) ** (m - 1) * np.exp(1j * n * rr[1]))
                + (R - rr[0])
                * np.real((-R + rr[0] + 1j * rr[2]) ** m * np.exp(1j * n * rr[1]))
            )
            * np.exp(
                (-(R**2) + 2 * R * rr[0] - rr[0] ** 2 - rr[2] ** 2) / (2 * d**2)
            )
            / (2 * np.sqrt(np.pi) * d**3 * rr[0]),
        ]
    )


@njit
def pert_maxwellboltzmann_dBdX(rr, R, d, m, n):
    return np.array(
        [
            np.sqrt(2)
            * (
                -(d**2)
                * rr[0]
                * (
                    d**2
                    * m
                    * (
                        2
                        * rr[2]
                        * np.real(
                            (-R + rr[0] + 1j * rr[2]) ** (m - 1)
                            * np.exp(1j * n * rr[1])
                        )
                        + 2
                        * (R - rr[0])
                        * np.imag(
                            (-R + rr[0] + 1j * rr[2]) ** (m - 1)
                            * np.exp(1j * n * rr[1])
                        )
                        - (m - 1)
                        * (rr[2] ** 2 + (R - rr[0]) ** 2)
                        * np.imag(
                            (-R + rr[0] + 1j * rr[2]) ** (m - 2)
                            * np.exp(1j * n * rr[1])
                        )
                    )
                    - m
                    * rr[2]
                    * (rr[2] ** 2 + (R - rr[0]) ** 2)
                    * np.real(
                        (-R + rr[0] + 1j * rr[2]) ** (m - 1) * np.exp(1j * n * rr[1])
                    )
                    + 2
                    * rr[2]
                    * (R - rr[0])
                    * np.real((-R + rr[0] + 1j * rr[2]) ** m * np.exp(1j * n * rr[1]))
                )
                - d**2
                * (
                    d**2
                    * (
                        m
                        * (rr[2] ** 2 + (R - rr[0]) ** 2)
                        * np.imag(
                            (-R + rr[0] + 1j * rr[2]) ** (m - 1)
                            * np.exp(1j * n * rr[1])
                        )
                        - 2
                        * rr[2]
                        * np.real(
                            (-R + rr[0] + 1j * rr[2]) ** m * np.exp(1j * n * rr[1])
                        )
                    )
                    + rr[2]
                    * (rr[2] ** 2 + (R - rr[0]) ** 2)
                    * np.real((-R + rr[0] + 1j * rr[2]) ** m * np.exp(1j * n * rr[1]))
                )
                + rr[0]
                * (R - rr[0])
                * (
                    d**2
                    * (
                        m
                        * (rr[2] ** 2 + (R - rr[0]) ** 2)
                        * np.imag(
                            (-R + rr[0] + 1j * rr[2]) ** (m - 1)
                            * np.exp(1j * n * rr[1])
                        )
                        - 2
                        * rr[2]
                        * np.real(
                            (-R + rr[0] + 1j * rr[2]) ** m * np.exp(1j * n * rr[1])
                        )
                    )
                    + rr[2]
                    * (rr[2] ** 2 + (R - rr[0]) ** 2)
                    * np.real((-R + rr[0] + 1j * rr[2]) ** m * np.exp(1j * n * rr[1]))
                )
            )
            * np.exp(
                (-(R**2) + 2 * R * rr[0] - rr[0] ** 2 - rr[2] ** 2) / (2 * d**2)
            )
            / (np.sqrt(np.pi) * d**7 * rr[0] ** 2),
            np.sqrt(2)
            * n
            * (
                d**2
                * (
                    m
                    * (rr[2] ** 2 + (R - rr[0]) ** 2)
                    * np.real(
                        (-R + rr[0] + 1j * rr[2]) ** (m - 1) * np.exp(1j * n * rr[1])
                    )
                    + 2
                    * rr[2]
                    * np.imag((-R + rr[0] + 1j * rr[2]) ** m * np.exp(1j * n * rr[1]))
                )
                - rr[2]
                * (rr[2] ** 2 + (R - rr[0]) ** 2)
                * np.imag((-R + rr[0] + 1j * rr[2]) ** m * np.exp(1j * n * rr[1]))
            )
            * np.exp(
                (-(R**2) + 2 * R * rr[0] - rr[0] ** 2 - rr[2] ** 2) / (2 * d**2)
            )
            / (np.sqrt(np.pi) * d**5 * rr[0] ** 2),
            np.sqrt(2)
            * (
                d**2
                * (
                    d**2
                    * (
                        4
                        * m
                        * rr[2]
                        * np.imag(
                            (-R + rr[0] + 1j * rr[2]) ** (m - 1)
                            * np.exp(1j * n * rr[1])
                        )
                        + m
                        * (m - 1)
                        * (rr[2] ** 2 + (R - rr[0]) ** 2)
                        * np.real(
                            (-R + rr[0] + 1j * rr[2]) ** (m - 2)
                            * np.exp(1j * n * rr[1])
                        )
                        - 2
                        * np.real(
                            (-R + rr[0] + 1j * rr[2]) ** m * np.exp(1j * n * rr[1])
                        )
                    )
                    - m
                    * rr[2]
                    * (rr[2] ** 2 + (R - rr[0]) ** 2)
                    * np.imag(
                        (-R + rr[0] + 1j * rr[2]) ** (m - 1) * np.exp(1j * n * rr[1])
                    )
                    + 2
                    * rr[2] ** 2
                    * np.real((-R + rr[0] + 1j * rr[2]) ** m * np.exp(1j * n * rr[1]))
                    + (rr[2] ** 2 + (R - rr[0]) ** 2)
                    * np.real((-R + rr[0] + 1j * rr[2]) ** m * np.exp(1j * n * rr[1]))
                )
                - rr[2]
                * (
                    d**2
                    * (
                        m
                        * (rr[2] ** 2 + (R - rr[0]) ** 2)
                        * np.imag(
                            (-R + rr[0] + 1j * rr[2]) ** (m - 1)
                            * np.exp(1j * n * rr[1])
                        )
                        - 2
                        * rr[2]
                        * np.real(
                            (-R + rr[0] + 1j * rr[2]) ** m * np.exp(1j * n * rr[1])
                        )
                    )
                    + rr[2]
                    * (rr[2] ** 2 + (R - rr[0]) ** 2)
                    * np.real((-R + rr[0] + 1j * rr[2]) ** m * np.exp(1j * n * rr[1]))
                )
            )
            * np.exp(
                (-(R**2) + 2 * R * rr[0] - rr[0] ** 2 - rr[2] ** 2) / (2 * d**2)
            )
            / (np.sqrt(np.pi) * d**7 * rr[0]),
            0,
            0,
            0,
            np.sqrt(2)
            * (
                -(d**2)
                * rr[0]
                * (
                    d**2
                    * (
                        4
                        * m
                        * (R - rr[0])
                        * np.real(
                            (-R + rr[0] + 1j * rr[2]) ** (m - 1)
                            * np.exp(1j * n * rr[1])
                        )
                        - m
                        * (m - 1)
                        * (rr[2] ** 2 + (R - rr[0]) ** 2)
                        * np.real(
                            (-R + rr[0] + 1j * rr[2]) ** (m - 2)
                            * np.exp(1j * n * rr[1])
                        )
                        - 2
                        * np.real(
                            (-R + rr[0] + 1j * rr[2]) ** m * np.exp(1j * n * rr[1])
                        )
                    )
                    - m
                    * (R - rr[0])
                    * (rr[2] ** 2 + (R - rr[0]) ** 2)
                    * np.real(
                        (-R + rr[0] + 1j * rr[2]) ** (m - 1) * np.exp(1j * n * rr[1])
                    )
                    + 2
                    * (R - rr[0]) ** 2
                    * np.real((-R + rr[0] + 1j * rr[2]) ** m * np.exp(1j * n * rr[1]))
                    + (rr[2] ** 2 + (R - rr[0]) ** 2)
                    * np.real((-R + rr[0] + 1j * rr[2]) ** m * np.exp(1j * n * rr[1]))
                )
                - d**2
                * (
                    d**2
                    * (
                        m
                        * (rr[2] ** 2 + (R - rr[0]) ** 2)
                        * np.real(
                            (-R + rr[0] + 1j * rr[2]) ** (m - 1)
                            * np.exp(1j * n * rr[1])
                        )
                        - 2
                        * (R - rr[0])
                        * np.real(
                            (-R + rr[0] + 1j * rr[2]) ** m * np.exp(1j * n * rr[1])
                        )
                    )
                    + (R - rr[0])
                    * (rr[2] ** 2 + (R - rr[0]) ** 2)
                    * np.real((-R + rr[0] + 1j * rr[2]) ** m * np.exp(1j * n * rr[1]))
                )
                + rr[0]
                * (R - rr[0])
                * (
                    d**2
                    * (
                        m
                        * (rr[2] ** 2 + (R - rr[0]) ** 2)
                        * np.real(
                            (-R + rr[0] + 1j * rr[2]) ** (m - 1)
                            * np.exp(1j * n * rr[1])
                        )
                        - 2
                        * (R - rr[0])
                        * np.real(
                            (-R + rr[0] + 1j * rr[2]) ** m * np.exp(1j * n * rr[1])
                        )
                    )
                    + (R - rr[0])
                    * (rr[2] ** 2 + (R - rr[0]) ** 2)
                    * np.real((-R + rr[0] + 1j * rr[2]) ** m * np.exp(1j * n * rr[1]))
                )
            )
            * np.exp(
                (-(R**2) + 2 * R * rr[0] - rr[0] ** 2 - rr[2] ** 2) / (2 * d**2)
            )
            / (np.sqrt(np.pi) * d**7 * rr[0] ** 2),
            np.sqrt(2)
            * n
            * (
                d**2
                * (
                    -m
                    * (rr[2] ** 2 + (R - rr[0]) ** 2)
                    * np.imag(
                        (-R + rr[0] + 1j * rr[2]) ** (m - 1) * np.exp(1j * n * rr[1])
                    )
                    + 2
                    * (R - rr[0])
                    * np.imag((-R + rr[0] + 1j * rr[2]) ** m * np.exp(1j * n * rr[1]))
                )
                - (R - rr[0])
                * (rr[2] ** 2 + (R - rr[0]) ** 2)
                * np.imag((-R + rr[0] + 1j * rr[2]) ** m * np.exp(1j * n * rr[1]))
            )
            * np.exp(
                (-(R**2) + 2 * R * rr[0] - rr[0] ** 2 - rr[2] ** 2) / (2 * d**2)
            )
            / (np.sqrt(np.pi) * d**5 * rr[0] ** 2),
            np.sqrt(2)
            * (
                d**2
                * (
                    d**2
                    * m
                    * (
                        2
                        * rr[2]
                        * np.real(
                            (-R + rr[0] + 1j * rr[2]) ** (m - 1)
                            * np.exp(1j * n * rr[1])
                        )
                        + 2
                        * (R - rr[0])
                        * np.imag(
                            (-R + rr[0] + 1j * rr[2]) ** (m - 1)
                            * np.exp(1j * n * rr[1])
                        )
                        - (m - 1)
                        * (rr[2] ** 2 + (R - rr[0]) ** 2)
                        * np.imag(
                            (-R + rr[0] + 1j * rr[2]) ** (m - 2)
                            * np.exp(1j * n * rr[1])
                        )
                    )
                    - m
                    * (R - rr[0])
                    * (rr[2] ** 2 + (R - rr[0]) ** 2)
                    * np.imag(
                        (-R + rr[0] + 1j * rr[2]) ** (m - 1) * np.exp(1j * n * rr[1])
                    )
                    + 2
                    * rr[2]
                    * (R - rr[0])
                    * np.real((-R + rr[0] + 1j * rr[2]) ** m * np.exp(1j * n * rr[1]))
                )
                - rr[2]
                * (
                    d**2
                    * (
                        m
                        * (rr[2] ** 2 + (R - rr[0]) ** 2)
                        * np.real(
                            (-R + rr[0] + 1j * rr[2]) ** (m - 1)
                            * np.exp(1j * n * rr[1])
                        )
                        - 2
                        * (R - rr[0])
                        * np.real(
                            (-R + rr[0] + 1j * rr[2]) ** m * np.exp(1j * n * rr[1])
                        )
                    )
                    + (R - rr[0])
                    * (rr[2] ** 2 + (R - rr[0]) ** 2)
                    * np.real((-R + rr[0] + 1j * rr[2]) ** m * np.exp(1j * n * rr[1]))
                )
            )
            * np.exp(
                (-(R**2) + 2 * R * rr[0] - rr[0] ** 2 - rr[2] ** 2) / (2 * d**2)
            )
            / (np.sqrt(np.pi) * d**7 * rr[0]),
        ]
    )


@njit
def pert_gaussian_dBdX(rr, R, d, m, n):
    return np.array(
        [
            np.sqrt(2)
            * (
                d**2
                * m
                * rr[0]
                * (
                    d**2
                    * (m - 1)
                    * np.imag(
                        (-R + rr[0] + 1j * rr[2]) ** (m - 2) * np.exp(1j * n * rr[1])
                    )
                    + rr[2]
                    * np.real(
                        (-R + rr[0] + 1j * rr[2]) ** (m - 1) * np.exp(1j * n * rr[1])
                    )
                )
                - d**2
                * (
                    d**2
                    * m
                    * np.imag(
                        (-R + rr[0] + 1j * rr[2]) ** (m - 1) * np.exp(1j * n * rr[1])
                    )
                    + rr[2]
                    * np.real((-R + rr[0] + 1j * rr[2]) ** m * np.exp(1j * n * rr[1]))
                )
                + rr[0]
                * (R - rr[0])
                * (
                    d**2
                    * m
                    * np.imag(
                        (-R + rr[0] + 1j * rr[2]) ** (m - 1) * np.exp(1j * n * rr[1])
                    )
                    + rr[2]
                    * np.real((-R + rr[0] + 1j * rr[2]) ** m * np.exp(1j * n * rr[1]))
                )
            )
            * np.exp(
                (-(R**2) + 2 * R * rr[0] - rr[0] ** 2 - rr[2] ** 2) / (2 * d**2)
            )
            / (2 * np.sqrt(np.pi) * d**5 * rr[0] ** 2),
            np.sqrt(2)
            * n
            * (
                d**2
                * m
                * np.real((-R + rr[0] + 1j * rr[2]) ** (m - 1) * np.exp(1j * n * rr[1]))
                - rr[2]
                * np.imag((-R + rr[0] + 1j * rr[2]) ** m * np.exp(1j * n * rr[1]))
            )
            * np.exp(
                (-(R**2) + 2 * R * rr[0] - rr[0] ** 2 - rr[2] ** 2) / (2 * d**2)
            )
            / (2 * np.sqrt(np.pi) * d**3 * rr[0] ** 2),
            np.sqrt(2)
            * (
                d**2
                * (
                    d**2
                    * m
                    * (m - 1)
                    * np.real(
                        (-R + rr[0] + 1j * rr[2]) ** (m - 2) * np.exp(1j * n * rr[1])
                    )
                    - m
                    * rr[2]
                    * np.imag(
                        (-R + rr[0] + 1j * rr[2]) ** (m - 1) * np.exp(1j * n * rr[1])
                    )
                    + np.real((-R + rr[0] + 1j * rr[2]) ** m * np.exp(1j * n * rr[1]))
                )
                - rr[2]
                * (
                    d**2
                    * m
                    * np.imag(
                        (-R + rr[0] + 1j * rr[2]) ** (m - 1) * np.exp(1j * n * rr[1])
                    )
                    + rr[2]
                    * np.real((-R + rr[0] + 1j * rr[2]) ** m * np.exp(1j * n * rr[1]))
                )
            )
            * np.exp(
                (-(R**2) + 2 * R * rr[0] - rr[0] ** 2 - rr[2] ** 2) / (2 * d**2)
            )
            / (2 * np.sqrt(np.pi) * d**5 * rr[0]),
            0,
            0,
            0,
            np.sqrt(2)
            * (
                d**2
                * rr[0]
                * (
                    d**2
                    * m
                    * (m - 1)
                    * np.real(
                        (-R + rr[0] + 1j * rr[2]) ** (m - 2) * np.exp(1j * n * rr[1])
                    )
                    + m
                    * (R - rr[0])
                    * np.real(
                        (-R + rr[0] + 1j * rr[2]) ** (m - 1) * np.exp(1j * n * rr[1])
                    )
                    - np.real((-R + rr[0] + 1j * rr[2]) ** m * np.exp(1j * n * rr[1]))
                )
                - d**2
                * (
                    d**2
                    * m
                    * np.real(
                        (-R + rr[0] + 1j * rr[2]) ** (m - 1) * np.exp(1j * n * rr[1])
                    )
                    + (R - rr[0])
                    * np.real((-R + rr[0] + 1j * rr[2]) ** m * np.exp(1j * n * rr[1]))
                )
                + rr[0]
                * (R - rr[0])
                * (
                    d**2
                    * m
                    * np.real(
                        (-R + rr[0] + 1j * rr[2]) ** (m - 1) * np.exp(1j * n * rr[1])
                    )
                    + (R - rr[0])
                    * np.real((-R + rr[0] + 1j * rr[2]) ** m * np.exp(1j * n * rr[1]))
                )
            )
            * np.exp(
                (-(R**2) + 2 * R * rr[0] - rr[0] ** 2 - rr[2] ** 2) / (2 * d**2)
            )
            / (2 * np.sqrt(np.pi) * d**5 * rr[0] ** 2),
            np.sqrt(2)
            * n
            * (
                -(d**2)
                * m
                * np.imag((-R + rr[0] + 1j * rr[2]) ** (m - 1) * np.exp(1j * n * rr[1]))
                - (R - rr[0])
                * np.imag((-R + rr[0] + 1j * rr[2]) ** m * np.exp(1j * n * rr[1]))
            )
            * np.exp(
                (-(R**2) + 2 * R * rr[0] - rr[0] ** 2 - rr[2] ** 2) / (2 * d**2)
            )
            / (2 * np.sqrt(np.pi) * d**3 * rr[0] ** 2),
            np.sqrt(2)
            * (
                -(d**2)
                * m
                * (
                    d**2
                    * (m - 1)
                    * np.imag(
                        (-R + rr[0] + 1j * rr[2]) ** (m - 2) * np.exp(1j * n * rr[1])
                    )
                    + (R - rr[0])
                    * np.imag(
                        (-R + rr[0] + 1j * rr[2]) ** (m - 1) * np.exp(1j * n * rr[1])
                    )
                )
                - rr[2]
                * (
                    d**2
                    * m
                    * np.real(
                        (-R + rr[0] + 1j * rr[2]) ** (m - 1) * np.exp(1j * n * rr[1])
                    )
                    + (R - rr[0])
                    * np.real((-R + rr[0] + 1j * rr[2]) ** m * np.exp(1j * n * rr[1]))
                )
            )
            * np.exp(
                (-(R**2) + 2 * R * rr[0] - rr[0] ** 2 - rr[2] ** 2) / (2 * d**2)
            )
            / (2 * np.sqrt(np.pi) * d**5 * rr[0]),
        ]
    )
