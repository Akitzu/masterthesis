import numpy as np
from numba import njit


@njit
def equ_squared(rr, shear, sf, R):
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
    )


@njit
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
                        * np.im(
                            (-R + rr[0] + 1j * rr[2]) ** (m - 1)
                            * np.exp(1j * n * rr[1])
                        )
                        - 2
                        * rr[2]
                        * np.re((-R + rr[0] + 1j * rr[2]) ** m * np.exp(1j * n * rr[1]))
                    )
                    + rr[2]
                    * (rr[2] ** 2 + (R - rr[0]) ** 2)
                    * np.re((-R + rr[0] + 1j * rr[2]) ** m * np.exp(1j * n * rr[1]))
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
                        * np.re(
                            (-R + rr[0] + 1j * rr[2]) ** (m - 1)
                            * np.exp(1j * n * rr[1])
                        )
                        - 2
                        * (R - rr[0])
                        * np.re((-R + rr[0] + 1j * rr[2]) ** m * np.exp(1j * n * rr[1]))
                    )
                    + (R - rr[0])
                    * (rr[2] ** 2 + (R - rr[0]) ** 2)
                    * np.re((-R + rr[0] + 1j * rr[2]) ** m * np.exp(1j * n * rr[1]))
                )
                * np.exp(
                    (-(R**2) + 2 * R * rr[0] - rr[0] ** 2 - rr[2] ** 2) / (2 * d**2)
                )
                / (np.sqrt(np.pi) * d**5 * rr[0])
            ],
        ]
    )


@njit
def pert_gaussian(rr, R, d, m, n):
    return np.array(
        [
            [
                np.sqrt(2)
                * (
                    d**2
                    * m
                    * np.im(
                        (-R + rr[0] + 1j * rr[2]) ** (m - 1) * np.exp(1j * n * rr[1])
                    )
                    + rr[2]
                    * np.re((-R + rr[0] + 1j * rr[2]) ** m * np.exp(1j * n * rr[1]))
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
                    * np.re(
                        (-R + rr[0] + 1j * rr[2]) ** (m - 1) * np.exp(1j * n * rr[1])
                    )
                    + (R - rr[0])
                    * np.re((-R + rr[0] + 1j * rr[2]) ** m * np.exp(1j * n * rr[1]))
                )
                * np.exp(
                    (-(R**2) + 2 * R * rr[0] - rr[0] ** 2 - rr[2] ** 2) / (2 * d**2)
                )
                / (2 * np.sqrt(np.pi) * d**3 * rr[0])
            ],
        ]
    )


@njit
def equ_squared_dBdX(rr, shear, sf, R):
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


@njit
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
                            * np.re(
                                (-R + rr[0] + 1j * rr[2]) ** (m - 1)
                                * np.exp(1j * n * rr[1])
                            )
                            + 2
                            * (R - rr[0])
                            * np.im(
                                (-R + rr[0] + 1j * rr[2]) ** (m - 1)
                                * np.exp(1j * n * rr[1])
                            )
                            - (m - 1)
                            * (rr[2] ** 2 + (R - rr[0]) ** 2)
                            * np.im(
                                (-R + rr[0] + 1j * rr[2]) ** (m - 2)
                                * np.exp(1j * n * rr[1])
                            )
                        )
                        - m
                        * rr[2]
                        * (rr[2] ** 2 + (R - rr[0]) ** 2)
                        * np.re(
                            (-R + rr[0] + 1j * rr[2]) ** (m - 1)
                            * np.exp(1j * n * rr[1])
                        )
                        + 2
                        * rr[2]
                        * (R - rr[0])
                        * np.re((-R + rr[0] + 1j * rr[2]) ** m * np.exp(1j * n * rr[1]))
                    )
                    - d**2
                    * (
                        d**2
                        * (
                            m
                            * (rr[2] ** 2 + (R - rr[0]) ** 2)
                            * np.im(
                                (-R + rr[0] + 1j * rr[2]) ** (m - 1)
                                * np.exp(1j * n * rr[1])
                            )
                            - 2
                            * rr[2]
                            * np.re(
                                (-R + rr[0] + 1j * rr[2]) ** m * np.exp(1j * n * rr[1])
                            )
                        )
                        + rr[2]
                        * (rr[2] ** 2 + (R - rr[0]) ** 2)
                        * np.re((-R + rr[0] + 1j * rr[2]) ** m * np.exp(1j * n * rr[1]))
                    )
                    + rr[0]
                    * (R - rr[0])
                    * (
                        d**2
                        * (
                            m
                            * (rr[2] ** 2 + (R - rr[0]) ** 2)
                            * np.im(
                                (-R + rr[0] + 1j * rr[2]) ** (m - 1)
                                * np.exp(1j * n * rr[1])
                            )
                            - 2
                            * rr[2]
                            * np.re(
                                (-R + rr[0] + 1j * rr[2]) ** m * np.exp(1j * n * rr[1])
                            )
                        )
                        + rr[2]
                        * (rr[2] ** 2 + (R - rr[0]) ** 2)
                        * np.re((-R + rr[0] + 1j * rr[2]) ** m * np.exp(1j * n * rr[1]))
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
                        * np.re(
                            (-R + rr[0] + 1j * rr[2]) ** (m - 1)
                            * np.exp(1j * n * rr[1])
                        )
                        + 2
                        * rr[2]
                        * np.im((-R + rr[0] + 1j * rr[2]) ** m * np.exp(1j * n * rr[1]))
                    )
                    - rr[2]
                    * (rr[2] ** 2 + (R - rr[0]) ** 2)
                    * np.im((-R + rr[0] + 1j * rr[2]) ** m * np.exp(1j * n * rr[1]))
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
                            * np.im(
                                (-R + rr[0] + 1j * rr[2]) ** (m - 1)
                                * np.exp(1j * n * rr[1])
                            )
                            + m
                            * (m - 1)
                            * (rr[2] ** 2 + (R - rr[0]) ** 2)
                            * np.re(
                                (-R + rr[0] + 1j * rr[2]) ** (m - 2)
                                * np.exp(1j * n * rr[1])
                            )
                            - 2
                            * np.re(
                                (-R + rr[0] + 1j * rr[2]) ** m * np.exp(1j * n * rr[1])
                            )
                        )
                        - m
                        * rr[2]
                        * (rr[2] ** 2 + (R - rr[0]) ** 2)
                        * np.im(
                            (-R + rr[0] + 1j * rr[2]) ** (m - 1)
                            * np.exp(1j * n * rr[1])
                        )
                        + 2
                        * rr[2] ** 2
                        * np.re((-R + rr[0] + 1j * rr[2]) ** m * np.exp(1j * n * rr[1]))
                        + (rr[2] ** 2 + (R - rr[0]) ** 2)
                        * np.re((-R + rr[0] + 1j * rr[2]) ** m * np.exp(1j * n * rr[1]))
                    )
                    - rr[2]
                    * (
                        d**2
                        * (
                            m
                            * (rr[2] ** 2 + (R - rr[0]) ** 2)
                            * np.im(
                                (-R + rr[0] + 1j * rr[2]) ** (m - 1)
                                * np.exp(1j * n * rr[1])
                            )
                            - 2
                            * rr[2]
                            * np.re(
                                (-R + rr[0] + 1j * rr[2]) ** m * np.exp(1j * n * rr[1])
                            )
                        )
                        + rr[2]
                        * (rr[2] ** 2 + (R - rr[0]) ** 2)
                        * np.re((-R + rr[0] + 1j * rr[2]) ** m * np.exp(1j * n * rr[1]))
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
                            * np.re(
                                (-R + rr[0] + 1j * rr[2]) ** (m - 1)
                                * np.exp(1j * n * rr[1])
                            )
                            - m
                            * (m - 1)
                            * (rr[2] ** 2 + (R - rr[0]) ** 2)
                            * np.re(
                                (-R + rr[0] + 1j * rr[2]) ** (m - 2)
                                * np.exp(1j * n * rr[1])
                            )
                            - 2
                            * np.re(
                                (-R + rr[0] + 1j * rr[2]) ** m * np.exp(1j * n * rr[1])
                            )
                        )
                        - m
                        * (R - rr[0])
                        * (rr[2] ** 2 + (R - rr[0]) ** 2)
                        * np.re(
                            (-R + rr[0] + 1j * rr[2]) ** (m - 1)
                            * np.exp(1j * n * rr[1])
                        )
                        + 2
                        * (R - rr[0]) ** 2
                        * np.re((-R + rr[0] + 1j * rr[2]) ** m * np.exp(1j * n * rr[1]))
                        + (rr[2] ** 2 + (R - rr[0]) ** 2)
                        * np.re((-R + rr[0] + 1j * rr[2]) ** m * np.exp(1j * n * rr[1]))
                    )
                    - d**2
                    * (
                        d**2
                        * (
                            m
                            * (rr[2] ** 2 + (R - rr[0]) ** 2)
                            * np.re(
                                (-R + rr[0] + 1j * rr[2]) ** (m - 1)
                                * np.exp(1j * n * rr[1])
                            )
                            - 2
                            * (R - rr[0])
                            * np.re(
                                (-R + rr[0] + 1j * rr[2]) ** m * np.exp(1j * n * rr[1])
                            )
                        )
                        + (R - rr[0])
                        * (rr[2] ** 2 + (R - rr[0]) ** 2)
                        * np.re((-R + rr[0] + 1j * rr[2]) ** m * np.exp(1j * n * rr[1]))
                    )
                    + rr[0]
                    * (R - rr[0])
                    * (
                        d**2
                        * (
                            m
                            * (rr[2] ** 2 + (R - rr[0]) ** 2)
                            * np.re(
                                (-R + rr[0] + 1j * rr[2]) ** (m - 1)
                                * np.exp(1j * n * rr[1])
                            )
                            - 2
                            * (R - rr[0])
                            * np.re(
                                (-R + rr[0] + 1j * rr[2]) ** m * np.exp(1j * n * rr[1])
                            )
                        )
                        + (R - rr[0])
                        * (rr[2] ** 2 + (R - rr[0]) ** 2)
                        * np.re((-R + rr[0] + 1j * rr[2]) ** m * np.exp(1j * n * rr[1]))
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
                        * np.im(
                            (-R + rr[0] + 1j * rr[2]) ** (m - 1)
                            * np.exp(1j * n * rr[1])
                        )
                        + 2
                        * (R - rr[0])
                        * np.im((-R + rr[0] + 1j * rr[2]) ** m * np.exp(1j * n * rr[1]))
                    )
                    - (R - rr[0])
                    * (rr[2] ** 2 + (R - rr[0]) ** 2)
                    * np.im((-R + rr[0] + 1j * rr[2]) ** m * np.exp(1j * n * rr[1]))
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
                            * np.re(
                                (-R + rr[0] + 1j * rr[2]) ** (m - 1)
                                * np.exp(1j * n * rr[1])
                            )
                            + 2
                            * (R - rr[0])
                            * np.im(
                                (-R + rr[0] + 1j * rr[2]) ** (m - 1)
                                * np.exp(1j * n * rr[1])
                            )
                            - (m - 1)
                            * (rr[2] ** 2 + (R - rr[0]) ** 2)
                            * np.im(
                                (-R + rr[0] + 1j * rr[2]) ** (m - 2)
                                * np.exp(1j * n * rr[1])
                            )
                        )
                        - m
                        * (R - rr[0])
                        * (rr[2] ** 2 + (R - rr[0]) ** 2)
                        * np.im(
                            (-R + rr[0] + 1j * rr[2]) ** (m - 1)
                            * np.exp(1j * n * rr[1])
                        )
                        + 2
                        * rr[2]
                        * (R - rr[0])
                        * np.re((-R + rr[0] + 1j * rr[2]) ** m * np.exp(1j * n * rr[1]))
                    )
                    - rr[2]
                    * (
                        d**2
                        * (
                            m
                            * (rr[2] ** 2 + (R - rr[0]) ** 2)
                            * np.re(
                                (-R + rr[0] + 1j * rr[2]) ** (m - 1)
                                * np.exp(1j * n * rr[1])
                            )
                            - 2
                            * (R - rr[0])
                            * np.re(
                                (-R + rr[0] + 1j * rr[2]) ** m * np.exp(1j * n * rr[1])
                            )
                        )
                        + (R - rr[0])
                        * (rr[2] ** 2 + (R - rr[0]) ** 2)
                        * np.re((-R + rr[0] + 1j * rr[2]) ** m * np.exp(1j * n * rr[1]))
                    )
                )
                * np.exp(
                    (-(R**2) + 2 * R * rr[0] - rr[0] ** 2 - rr[2] ** 2) / (2 * d**2)
                )
                / (np.sqrt(np.pi) * d**7 * rr[0]),
            ],
        ]
    )


@njit
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
                        * np.im(
                            (-R + rr[0] + 1j * rr[2]) ** (m - 2)
                            * np.exp(1j * n * rr[1])
                        )
                        + rr[2]
                        * np.re(
                            (-R + rr[0] + 1j * rr[2]) ** (m - 1)
                            * np.exp(1j * n * rr[1])
                        )
                    )
                    - d**2
                    * (
                        d**2
                        * m
                        * np.im(
                            (-R + rr[0] + 1j * rr[2]) ** (m - 1)
                            * np.exp(1j * n * rr[1])
                        )
                        + rr[2]
                        * np.re((-R + rr[0] + 1j * rr[2]) ** m * np.exp(1j * n * rr[1]))
                    )
                    + rr[0]
                    * (R - rr[0])
                    * (
                        d**2
                        * m
                        * np.im(
                            (-R + rr[0] + 1j * rr[2]) ** (m - 1)
                            * np.exp(1j * n * rr[1])
                        )
                        + rr[2]
                        * np.re((-R + rr[0] + 1j * rr[2]) ** m * np.exp(1j * n * rr[1]))
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
                    * np.re(
                        (-R + rr[0] + 1j * rr[2]) ** (m - 1) * np.exp(1j * n * rr[1])
                    )
                    - rr[2]
                    * np.im((-R + rr[0] + 1j * rr[2]) ** m * np.exp(1j * n * rr[1]))
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
                        * np.re(
                            (-R + rr[0] + 1j * rr[2]) ** (m - 2)
                            * np.exp(1j * n * rr[1])
                        )
                        - m
                        * rr[2]
                        * np.im(
                            (-R + rr[0] + 1j * rr[2]) ** (m - 1)
                            * np.exp(1j * n * rr[1])
                        )
                        + np.re((-R + rr[0] + 1j * rr[2]) ** m * np.exp(1j * n * rr[1]))
                    )
                    - rr[2]
                    * (
                        d**2
                        * m
                        * np.im(
                            (-R + rr[0] + 1j * rr[2]) ** (m - 1)
                            * np.exp(1j * n * rr[1])
                        )
                        + rr[2]
                        * np.re((-R + rr[0] + 1j * rr[2]) ** m * np.exp(1j * n * rr[1]))
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
                        * np.re(
                            (-R + rr[0] + 1j * rr[2]) ** (m - 2)
                            * np.exp(1j * n * rr[1])
                        )
                        + m
                        * (R - rr[0])
                        * np.re(
                            (-R + rr[0] + 1j * rr[2]) ** (m - 1)
                            * np.exp(1j * n * rr[1])
                        )
                        - np.re((-R + rr[0] + 1j * rr[2]) ** m * np.exp(1j * n * rr[1]))
                    )
                    - d**2
                    * (
                        d**2
                        * m
                        * np.re(
                            (-R + rr[0] + 1j * rr[2]) ** (m - 1)
                            * np.exp(1j * n * rr[1])
                        )
                        + (R - rr[0])
                        * np.re((-R + rr[0] + 1j * rr[2]) ** m * np.exp(1j * n * rr[1]))
                    )
                    + rr[0]
                    * (R - rr[0])
                    * (
                        d**2
                        * m
                        * np.re(
                            (-R + rr[0] + 1j * rr[2]) ** (m - 1)
                            * np.exp(1j * n * rr[1])
                        )
                        + (R - rr[0])
                        * np.re((-R + rr[0] + 1j * rr[2]) ** m * np.exp(1j * n * rr[1]))
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
                    * np.im(
                        (-R + rr[0] + 1j * rr[2]) ** (m - 1) * np.exp(1j * n * rr[1])
                    )
                    - (R - rr[0])
                    * np.im((-R + rr[0] + 1j * rr[2]) ** m * np.exp(1j * n * rr[1]))
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
                        * np.im(
                            (-R + rr[0] + 1j * rr[2]) ** (m - 2)
                            * np.exp(1j * n * rr[1])
                        )
                        + (R - rr[0])
                        * np.im(
                            (-R + rr[0] + 1j * rr[2]) ** (m - 1)
                            * np.exp(1j * n * rr[1])
                        )
                    )
                    - rr[2]
                    * (
                        d**2
                        * m
                        * np.re(
                            (-R + rr[0] + 1j * rr[2]) ** (m - 1)
                            * np.exp(1j * n * rr[1])
                        )
                        + (R - rr[0])
                        * np.re((-R + rr[0] + 1j * rr[2]) ** m * np.exp(1j * n * rr[1]))
                    )
                )
                * np.exp(
                    (-(R**2) + 2 * R * rr[0] - rr[0] ** 2 - rr[2] ** 2) / (2 * d**2)
                )
                / (2 * np.sqrt(np.pi) * d**5 * rr[0]),
            ],
        ]
    )
