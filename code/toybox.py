import numpy as np
from numba import jit
import functools

## Usefull decorators


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


## Equilibrium field


def equ_squared(rr, R, sf, shear):
    return np.array(
        [
            [-2 * rr[2] / rr[0]],
            [
                (2 * sf + 2 * shear * (rr[2] ** 2 + (R - rr[0]) ** 2))
                * np.sqrt(R**2 - rr[2] ** 2 - (R - rr[0]) ** 2)
                / rr[0]
            ],
            [(-2 * R + 2 * rr[0]) / rr[0]],
        ]
    ).squeeze()


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


def pert_maxwellboltzmann(rr, R, d, m, n):
    return np.array(
        [
            [
                np.sqrt(2)
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
                * np.exp(
                    (-(R**2) + 2 * R * rr[0] - rr[0] ** 2 - rr[2] ** 2) / (2 * d**2)
                )
                / (np.sqrt(np.pi) * d**5 * rr[0])
            ],
            [0],
            [
                np.sqrt(2)
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
                * np.exp(
                    (-(R**2) + 2 * R * rr[0] - rr[0] ** 2 - rr[2] ** 2) / (2 * d**2)
                )
                / (np.sqrt(np.pi) * d**5 * rr[0])
            ],
        ]
    ).squeeze()


def pert_gaussian(rr, R, d, m, n):
    return np.array(
        [
            [
                np.sqrt(2)
                * (
                    d**2
                    * m
                    * np.imag(
                        (-R + rr[0] + 1j * rr[2]) ** (m - 1) * np.exp(1j * n * rr[1])
                    )
                    + rr[2]
                    * np.real((-R + rr[0] + 1j * rr[2]) ** m * np.exp(1j * n * rr[1]))
                )
                * np.exp(
                    (-(R**2) + 2 * R * rr[0] - rr[0] ** 2 - rr[2] ** 2) / (2 * d**2)
                )
                / (2 * np.sqrt(np.pi) * d**3 * rr[0])
            ],
            [0],
            [
                np.sqrt(2)
                * (
                    d**2
                    * m
                    * np.real(
                        (-R + rr[0] + 1j * rr[2]) ** (m - 1) * np.exp(1j * n * rr[1])
                    )
                    + (R - rr[0])
                    * np.real((-R + rr[0] + 1j * rr[2]) ** m * np.exp(1j * n * rr[1]))
                )
                * np.exp(
                    (-(R**2) + 2 * R * rr[0] - rr[0] ** 2 - rr[2] ** 2) / (2 * d**2)
                )
                / (2 * np.sqrt(np.pi) * d**3 * rr[0])
            ],
        ]
    ).squeeze()


def pert_maxwellboltzmann_dBdX(rr, R, d, m, n):
    return np.array(
        [
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
                            (-R + rr[0] + 1j * rr[2]) ** (m - 1)
                            * np.exp(1j * n * rr[1])
                        )
                        + 2
                        * rr[2]
                        * (R - rr[0])
                        * np.real(
                            (-R + rr[0] + 1j * rr[2]) ** m * np.exp(1j * n * rr[1])
                        )
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
                        * np.real(
                            (-R + rr[0] + 1j * rr[2]) ** m * np.exp(1j * n * rr[1])
                        )
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
                        * np.real(
                            (-R + rr[0] + 1j * rr[2]) ** m * np.exp(1j * n * rr[1])
                        )
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
                            (-R + rr[0] + 1j * rr[2]) ** (m - 1)
                            * np.exp(1j * n * rr[1])
                        )
                        + 2
                        * rr[2]
                        * np.imag(
                            (-R + rr[0] + 1j * rr[2]) ** m * np.exp(1j * n * rr[1])
                        )
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
                            (-R + rr[0] + 1j * rr[2]) ** (m - 1)
                            * np.exp(1j * n * rr[1])
                        )
                        + 2
                        * rr[2] ** 2
                        * np.real(
                            (-R + rr[0] + 1j * rr[2]) ** m * np.exp(1j * n * rr[1])
                        )
                        + (rr[2] ** 2 + (R - rr[0]) ** 2)
                        * np.real(
                            (-R + rr[0] + 1j * rr[2]) ** m * np.exp(1j * n * rr[1])
                        )
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
                        * np.real(
                            (-R + rr[0] + 1j * rr[2]) ** m * np.exp(1j * n * rr[1])
                        )
                    )
                )
                * np.exp(
                    (-(R**2) + 2 * R * rr[0] - rr[0] ** 2 - rr[2] ** 2) / (2 * d**2)
                )
                / (np.sqrt(np.pi) * d**7 * rr[0]),
            ],
            [0, 0, 0],
            [
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
                            (-R + rr[0] + 1j * rr[2]) ** (m - 1)
                            * np.exp(1j * n * rr[1])
                        )
                        + 2
                        * (R - rr[0]) ** 2
                        * np.real(
                            (-R + rr[0] + 1j * rr[2]) ** m * np.exp(1j * n * rr[1])
                        )
                        + (rr[2] ** 2 + (R - rr[0]) ** 2)
                        * np.real(
                            (-R + rr[0] + 1j * rr[2]) ** m * np.exp(1j * n * rr[1])
                        )
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
                        * np.real(
                            (-R + rr[0] + 1j * rr[2]) ** m * np.exp(1j * n * rr[1])
                        )
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
                        * np.real(
                            (-R + rr[0] + 1j * rr[2]) ** m * np.exp(1j * n * rr[1])
                        )
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
                            (-R + rr[0] + 1j * rr[2]) ** (m - 1)
                            * np.exp(1j * n * rr[1])
                        )
                        + 2
                        * (R - rr[0])
                        * np.imag(
                            (-R + rr[0] + 1j * rr[2]) ** m * np.exp(1j * n * rr[1])
                        )
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
                            (-R + rr[0] + 1j * rr[2]) ** (m - 1)
                            * np.exp(1j * n * rr[1])
                        )
                        + 2
                        * rr[2]
                        * (R - rr[0])
                        * np.real(
                            (-R + rr[0] + 1j * rr[2]) ** m * np.exp(1j * n * rr[1])
                        )
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
                        * np.real(
                            (-R + rr[0] + 1j * rr[2]) ** m * np.exp(1j * n * rr[1])
                        )
                    )
                )
                * np.exp(
                    (-(R**2) + 2 * R * rr[0] - rr[0] ** 2 - rr[2] ** 2) / (2 * d**2)
                )
                / (np.sqrt(np.pi) * d**7 * rr[0]),
            ],
        ]
    )


def pert_gaussian_dBdX(rr, R, d, m, n):
    return np.array(
        [
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
                            (-R + rr[0] + 1j * rr[2]) ** (m - 2)
                            * np.exp(1j * n * rr[1])
                        )
                        + rr[2]
                        * np.real(
                            (-R + rr[0] + 1j * rr[2]) ** (m - 1)
                            * np.exp(1j * n * rr[1])
                        )
                    )
                    - d**2
                    * (
                        d**2
                        * m
                        * np.imag(
                            (-R + rr[0] + 1j * rr[2]) ** (m - 1)
                            * np.exp(1j * n * rr[1])
                        )
                        + rr[2]
                        * np.real(
                            (-R + rr[0] + 1j * rr[2]) ** m * np.exp(1j * n * rr[1])
                        )
                    )
                    + rr[0]
                    * (R - rr[0])
                    * (
                        d**2
                        * m
                        * np.imag(
                            (-R + rr[0] + 1j * rr[2]) ** (m - 1)
                            * np.exp(1j * n * rr[1])
                        )
                        + rr[2]
                        * np.real(
                            (-R + rr[0] + 1j * rr[2]) ** m * np.exp(1j * n * rr[1])
                        )
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
                    * np.real(
                        (-R + rr[0] + 1j * rr[2]) ** (m - 1) * np.exp(1j * n * rr[1])
                    )
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
                            (-R + rr[0] + 1j * rr[2]) ** (m - 2)
                            * np.exp(1j * n * rr[1])
                        )
                        - m
                        * rr[2]
                        * np.imag(
                            (-R + rr[0] + 1j * rr[2]) ** (m - 1)
                            * np.exp(1j * n * rr[1])
                        )
                        + np.real(
                            (-R + rr[0] + 1j * rr[2]) ** m * np.exp(1j * n * rr[1])
                        )
                    )
                    - rr[2]
                    * (
                        d**2
                        * m
                        * np.imag(
                            (-R + rr[0] + 1j * rr[2]) ** (m - 1)
                            * np.exp(1j * n * rr[1])
                        )
                        + rr[2]
                        * np.real(
                            (-R + rr[0] + 1j * rr[2]) ** m * np.exp(1j * n * rr[1])
                        )
                    )
                )
                * np.exp(
                    (-(R**2) + 2 * R * rr[0] - rr[0] ** 2 - rr[2] ** 2) / (2 * d**2)
                )
                / (2 * np.sqrt(np.pi) * d**5 * rr[0]),
            ],
            [0, 0, 0],
            [
                np.sqrt(2)
                * (
                    d**2
                    * rr[0]
                    * (
                        d**2
                        * m
                        * (m - 1)
                        * np.real(
                            (-R + rr[0] + 1j * rr[2]) ** (m - 2)
                            * np.exp(1j * n * rr[1])
                        )
                        + m
                        * (R - rr[0])
                        * np.real(
                            (-R + rr[0] + 1j * rr[2]) ** (m - 1)
                            * np.exp(1j * n * rr[1])
                        )
                        - np.real(
                            (-R + rr[0] + 1j * rr[2]) ** m * np.exp(1j * n * rr[1])
                        )
                    )
                    - d**2
                    * (
                        d**2
                        * m
                        * np.real(
                            (-R + rr[0] + 1j * rr[2]) ** (m - 1)
                            * np.exp(1j * n * rr[1])
                        )
                        + (R - rr[0])
                        * np.real(
                            (-R + rr[0] + 1j * rr[2]) ** m * np.exp(1j * n * rr[1])
                        )
                    )
                    + rr[0]
                    * (R - rr[0])
                    * (
                        d**2
                        * m
                        * np.real(
                            (-R + rr[0] + 1j * rr[2]) ** (m - 1)
                            * np.exp(1j * n * rr[1])
                        )
                        + (R - rr[0])
                        * np.real(
                            (-R + rr[0] + 1j * rr[2]) ** m * np.exp(1j * n * rr[1])
                        )
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
                    * np.imag(
                        (-R + rr[0] + 1j * rr[2]) ** (m - 1) * np.exp(1j * n * rr[1])
                    )
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
                            (-R + rr[0] + 1j * rr[2]) ** (m - 2)
                            * np.exp(1j * n * rr[1])
                        )
                        + (R - rr[0])
                        * np.imag(
                            (-R + rr[0] + 1j * rr[2]) ** (m - 1)
                            * np.exp(1j * n * rr[1])
                        )
                    )
                    - rr[2]
                    * (
                        d**2
                        * m
                        * np.real(
                            (-R + rr[0] + 1j * rr[2]) ** (m - 1)
                            * np.exp(1j * n * rr[1])
                        )
                        + (R - rr[0])
                        * np.real(
                            (-R + rr[0] + 1j * rr[2]) ** m * np.exp(1j * n * rr[1])
                        )
                    )
                )
                * np.exp(
                    (-(R**2) + 2 * R * rr[0] - rr[0] ** 2 - rr[2] ** 2) / (2 * d**2)
                )
                / (2 * np.sqrt(np.pi) * d**5 * rr[0]),
            ],
        ]
    )
