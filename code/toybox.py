from pyoculus.problems import CylindricalBfield
from functools import partial
from jax import jit, jacfwd
import jax.numpy as jnp

# class AnalyticCylindricalBfield(CylindricalBfield):
#     def __init__(
#         self,
#         R,
#         sf,
#         shear,
#         perturbations_args
#     ):
#         # pert1_dict = {m:1, n:2, type: "maxwell-boltzmann", amplitude: 10}
#         # pert2_dict = {m:1, n:2, type: "gaussian", amplitude: 10}
#         # myBfield = AnalyticalBfieldProblem(R= 3, sf = 1.1, shear=3 pert=[pert1_dict, pert2_dict])
#         # myBfield.pert_list[0](rphiz)
#         # >> value

#         super().__init__(self.equilibrium_args["R"], 0)

#         self.B_equilibrium = partial(B_equilibrium[0], **self.equilibrium_args)
#         self.B_perturbation = partial(B_perturbation[0], **self.perturbation_args)
#         self.dBdX_equilibrium = partial(B_equilibrium[1], **self.equilibrium_args)
#         self.dBdX_perturbation = partial(B_perturbation[1], **self.perturbation_args)

#     @property
#     def amplitudes(self):
#         return [pert["amplitude"] for pert in self.perturbation_args]
    
#     @amplitudes.setter(self, value):


#     def B(self, rr):
#         return self.B_equilibrium(rr) + self._A_p * self.B_perturbation(rr)

#     def dBdX(self, rr):
#         return self.dBdX_equilibrium(rr) + self._A_p * self.dBdX_perturbation(rr)

#     def B_many(self, r, phi, z, input1D=True):
#         return jnp.array([self.B([r[i], phi[i], z[i]]) for i in range(len(r))])

#     def dBdX_many(self, r, phi, z, input1D=True):
#         return jnp.array([self.dBdX([r[i], phi[i], z[i]]) for i in range(len(r))])


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


@jit
def equ_squared_dBdX(rr, R, sf, shear):
    return jnp.array(
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
                / (rr[0] ** 2 * jnp.sqrt(R**2 - rr[2] ** 2 - (R - rr[0]) ** 2)),
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
                / (rr[0] * jnp.sqrt(2 * R * rr[0] - rr[0] ** 2 - rr[2] ** 2)),
            ],
            [2 * R / rr[0] ** 2, 0, 0],
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
                * jnp.imag((-R + rr[0] + 1j * rr[2]) ** (m - 1) * jnp.exp(1j * n * rr[1]))
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
                * jnp.real((-R + rr[0] + 1j * rr[2]) ** (m - 1) * jnp.exp(1j * n * rr[1]))
                + (R - rr[0])
                * jnp.real((-R + rr[0] + 1j * rr[2]) ** m * jnp.exp(1j * n * rr[1]))
            )
            * jnp.exp(
                (-(R**2) + 2 * R * rr[0] - rr[0] ** 2 - rr[2] ** 2) / (2 * d**2)
            )
            / (2 * jnp.sqrt(jnp.pi) * d**3 * rr[0]),
        ]
    )


@jit
def pert_maxwellboltzmann_dBdX(rr, R, d, m, n):
    return jnp.array(
        [
            jnp.sqrt(2)
            * (
                -(d**2)
                * rr[0]
                * (
                    d**2
                    * m
                    * (
                        2
                        * rr[2]
                        * jnp.real(
                            (-R + rr[0] + 1j * rr[2]) ** (m - 1)
                            * jnp.exp(1j * n * rr[1])
                        )
                        + 2
                        * (R - rr[0])
                        * jnp.imag(
                            (-R + rr[0] + 1j * rr[2]) ** (m - 1)
                            * jnp.exp(1j * n * rr[1])
                        )
                        - (m - 1)
                        * (rr[2] ** 2 + (R - rr[0]) ** 2)
                        * jnp.imag(
                            (-R + rr[0] + 1j * rr[2]) ** (m - 2)
                            * jnp.exp(1j * n * rr[1])
                        )
                    )
                    - m
                    * rr[2]
                    * (rr[2] ** 2 + (R - rr[0]) ** 2)
                    * jnp.real(
                        (-R + rr[0] + 1j * rr[2]) ** (m - 1) * jnp.exp(1j * n * rr[1])
                    )
                    + 2
                    * rr[2]
                    * (R - rr[0])
                    * jnp.real((-R + rr[0] + 1j * rr[2]) ** m * jnp.exp(1j * n * rr[1]))
                )
                - d**2
                * (
                    d**2
                    * (
                        m
                        * (rr[2] ** 2 + (R - rr[0]) ** 2)
                        * jnp.imag(
                            (-R + rr[0] + 1j * rr[2]) ** (m - 1)
                            * jnp.exp(1j * n * rr[1])
                        )
                        - 2
                        * rr[2]
                        * jnp.real(
                            (-R + rr[0] + 1j * rr[2]) ** m * jnp.exp(1j * n * rr[1])
                        )
                    )
                    + rr[2]
                    * (rr[2] ** 2 + (R - rr[0]) ** 2)
                    * jnp.real((-R + rr[0] + 1j * rr[2]) ** m * jnp.exp(1j * n * rr[1]))
                )
                + rr[0]
                * (R - rr[0])
                * (
                    d**2
                    * (
                        m
                        * (rr[2] ** 2 + (R - rr[0]) ** 2)
                        * jnp.imag(
                            (-R + rr[0] + 1j * rr[2]) ** (m - 1)
                            * jnp.exp(1j * n * rr[1])
                        )
                        - 2
                        * rr[2]
                        * jnp.real(
                            (-R + rr[0] + 1j * rr[2]) ** m * jnp.exp(1j * n * rr[1])
                        )
                    )
                    + rr[2]
                    * (rr[2] ** 2 + (R - rr[0]) ** 2)
                    * jnp.real((-R + rr[0] + 1j * rr[2]) ** m * jnp.exp(1j * n * rr[1]))
                )
            )
            * jnp.exp(
                (-(R**2) + 2 * R * rr[0] - rr[0] ** 2 - rr[2] ** 2) / (2 * d**2)
            )
            / (jnp.sqrt(jnp.pi) * d**7 * rr[0] ** 2),
            jnp.sqrt(2)
            * n
            * (
                d**2
                * (
                    m
                    * (rr[2] ** 2 + (R - rr[0]) ** 2)
                    * jnp.real(
                        (-R + rr[0] + 1j * rr[2]) ** (m - 1) * jnp.exp(1j * n * rr[1])
                    )
                    + 2
                    * rr[2]
                    * jnp.imag((-R + rr[0] + 1j * rr[2]) ** m * jnp.exp(1j * n * rr[1]))
                )
                - rr[2]
                * (rr[2] ** 2 + (R - rr[0]) ** 2)
                * jnp.imag((-R + rr[0] + 1j * rr[2]) ** m * jnp.exp(1j * n * rr[1]))
            )
            * jnp.exp(
                (-(R**2) + 2 * R * rr[0] - rr[0] ** 2 - rr[2] ** 2) / (2 * d**2)
            )
            / (jnp.sqrt(jnp.pi) * d**5 * rr[0] ** 2),
            jnp.sqrt(2)
            * (
                d**2
                * (
                    d**2
                    * (
                        4
                        * m
                        * rr[2]
                        * jnp.imag(
                            (-R + rr[0] + 1j * rr[2]) ** (m - 1)
                            * jnp.exp(1j * n * rr[1])
                        )
                        + m
                        * (m - 1)
                        * (rr[2] ** 2 + (R - rr[0]) ** 2)
                        * jnp.real(
                            (-R + rr[0] + 1j * rr[2]) ** (m - 2)
                            * jnp.exp(1j * n * rr[1])
                        )
                        - 2
                        * jnp.real(
                            (-R + rr[0] + 1j * rr[2]) ** m * jnp.exp(1j * n * rr[1])
                        )
                    )
                    - m
                    * rr[2]
                    * (rr[2] ** 2 + (R - rr[0]) ** 2)
                    * jnp.imag(
                        (-R + rr[0] + 1j * rr[2]) ** (m - 1) * jnp.exp(1j * n * rr[1])
                    )
                    + 2
                    * rr[2] ** 2
                    * jnp.real((-R + rr[0] + 1j * rr[2]) ** m * jnp.exp(1j * n * rr[1]))
                    + (rr[2] ** 2 + (R - rr[0]) ** 2)
                    * jnp.real((-R + rr[0] + 1j * rr[2]) ** m * jnp.exp(1j * n * rr[1]))
                )
                - rr[2]
                * (
                    d**2
                    * (
                        m
                        * (rr[2] ** 2 + (R - rr[0]) ** 2)
                        * jnp.imag(
                            (-R + rr[0] + 1j * rr[2]) ** (m - 1)
                            * jnp.exp(1j * n * rr[1])
                        )
                        - 2
                        * rr[2]
                        * jnp.real(
                            (-R + rr[0] + 1j * rr[2]) ** m * jnp.exp(1j * n * rr[1])
                        )
                    )
                    + rr[2]
                    * (rr[2] ** 2 + (R - rr[0]) ** 2)
                    * jnp.real((-R + rr[0] + 1j * rr[2]) ** m * jnp.exp(1j * n * rr[1]))
                )
            )
            * jnp.exp(
                (-(R**2) + 2 * R * rr[0] - rr[0] ** 2 - rr[2] ** 2) / (2 * d**2)
            )
            / (jnp.sqrt(jnp.pi) * d**7 * rr[0]),
            0,
            0,
            0,
            jnp.sqrt(2)
            * (
                -(d**2)
                * rr[0]
                * (
                    d**2
                    * (
                        4
                        * m
                        * (R - rr[0])
                        * jnp.real(
                            (-R + rr[0] + 1j * rr[2]) ** (m - 1)
                            * jnp.exp(1j * n * rr[1])
                        )
                        - m
                        * (m - 1)
                        * (rr[2] ** 2 + (R - rr[0]) ** 2)
                        * jnp.real(
                            (-R + rr[0] + 1j * rr[2]) ** (m - 2)
                            * jnp.exp(1j * n * rr[1])
                        )
                        - 2
                        * jnp.real(
                            (-R + rr[0] + 1j * rr[2]) ** m * jnp.exp(1j * n * rr[1])
                        )
                    )
                    - m
                    * (R - rr[0])
                    * (rr[2] ** 2 + (R - rr[0]) ** 2)
                    * jnp.real(
                        (-R + rr[0] + 1j * rr[2]) ** (m - 1) * jnp.exp(1j * n * rr[1])
                    )
                    + 2
                    * (R - rr[0]) ** 2
                    * jnp.real((-R + rr[0] + 1j * rr[2]) ** m * jnp.exp(1j * n * rr[1]))
                    + (rr[2] ** 2 + (R - rr[0]) ** 2)
                    * jnp.real((-R + rr[0] + 1j * rr[2]) ** m * jnp.exp(1j * n * rr[1]))
                )
                - d**2
                * (
                    d**2
                    * (
                        m
                        * (rr[2] ** 2 + (R - rr[0]) ** 2)
                        * jnp.real(
                            (-R + rr[0] + 1j * rr[2]) ** (m - 1)
                            * jnp.exp(1j * n * rr[1])
                        )
                        - 2
                        * (R - rr[0])
                        * jnp.real(
                            (-R + rr[0] + 1j * rr[2]) ** m * jnp.exp(1j * n * rr[1])
                        )
                    )
                    + (R - rr[0])
                    * (rr[2] ** 2 + (R - rr[0]) ** 2)
                    * jnp.real((-R + rr[0] + 1j * rr[2]) ** m * jnp.exp(1j * n * rr[1]))
                )
                + rr[0]
                * (R - rr[0])
                * (
                    d**2
                    * (
                        m
                        * (rr[2] ** 2 + (R - rr[0]) ** 2)
                        * jnp.real(
                            (-R + rr[0] + 1j * rr[2]) ** (m - 1)
                            * jnp.exp(1j * n * rr[1])
                        )
                        - 2
                        * (R - rr[0])
                        * jnp.real(
                            (-R + rr[0] + 1j * rr[2]) ** m * jnp.exp(1j * n * rr[1])
                        )
                    )
                    + (R - rr[0])
                    * (rr[2] ** 2 + (R - rr[0]) ** 2)
                    * jnp.real((-R + rr[0] + 1j * rr[2]) ** m * jnp.exp(1j * n * rr[1]))
                )
            )
            * jnp.exp(
                (-(R**2) + 2 * R * rr[0] - rr[0] ** 2 - rr[2] ** 2) / (2 * d**2)
            )
            / (jnp.sqrt(jnp.pi) * d**7 * rr[0] ** 2),
            jnp.sqrt(2)
            * n
            * (
                d**2
                * (
                    -m
                    * (rr[2] ** 2 + (R - rr[0]) ** 2)
                    * jnp.imag(
                        (-R + rr[0] + 1j * rr[2]) ** (m - 1) * jnp.exp(1j * n * rr[1])
                    )
                    + 2
                    * (R - rr[0])
                    * jnp.imag((-R + rr[0] + 1j * rr[2]) ** m * jnp.exp(1j * n * rr[1]))
                )
                - (R - rr[0])
                * (rr[2] ** 2 + (R - rr[0]) ** 2)
                * jnp.imag((-R + rr[0] + 1j * rr[2]) ** m * jnp.exp(1j * n * rr[1]))
            )
            * jnp.exp(
                (-(R**2) + 2 * R * rr[0] - rr[0] ** 2 - rr[2] ** 2) / (2 * d**2)
            )
            / (jnp.sqrt(jnp.pi) * d**5 * rr[0] ** 2),
            jnp.sqrt(2)
            * (
                d**2
                * (
                    d**2
                    * m
                    * (
                        2
                        * rr[2]
                        * jnp.real(
                            (-R + rr[0] + 1j * rr[2]) ** (m - 1)
                            * jnp.exp(1j * n * rr[1])
                        )
                        + 2
                        * (R - rr[0])
                        * jnp.imag(
                            (-R + rr[0] + 1j * rr[2]) ** (m - 1)
                            * jnp.exp(1j * n * rr[1])
                        )
                        - (m - 1)
                        * (rr[2] ** 2 + (R - rr[0]) ** 2)
                        * jnp.imag(
                            (-R + rr[0] + 1j * rr[2]) ** (m - 2)
                            * jnp.exp(1j * n * rr[1])
                        )
                    )
                    - m
                    * (R - rr[0])
                    * (rr[2] ** 2 + (R - rr[0]) ** 2)
                    * jnp.imag(
                        (-R + rr[0] + 1j * rr[2]) ** (m - 1) * jnp.exp(1j * n * rr[1])
                    )
                    + 2
                    * rr[2]
                    * (R - rr[0])
                    * jnp.real((-R + rr[0] + 1j * rr[2]) ** m * jnp.exp(1j * n * rr[1]))
                )
                - rr[2]
                * (
                    d**2
                    * (
                        m
                        * (rr[2] ** 2 + (R - rr[0]) ** 2)
                        * jnp.real(
                            (-R + rr[0] + 1j * rr[2]) ** (m - 1)
                            * jnp.exp(1j * n * rr[1])
                        )
                        - 2
                        * (R - rr[0])
                        * jnp.real(
                            (-R + rr[0] + 1j * rr[2]) ** m * jnp.exp(1j * n * rr[1])
                        )
                    )
                    + (R - rr[0])
                    * (rr[2] ** 2 + (R - rr[0]) ** 2)
                    * jnp.real((-R + rr[0] + 1j * rr[2]) ** m * jnp.exp(1j * n * rr[1]))
                )
            )
            * jnp.exp(
                (-(R**2) + 2 * R * rr[0] - rr[0] ** 2 - rr[2] ** 2) / (2 * d**2)
            )
            / (jnp.sqrt(jnp.pi) * d**7 * rr[0]),
        ]
    )


@jit
def pert_gaussian_dBdX(rr, R, d, m, n):
    return jnp.array(
        [
            jnp.sqrt(2)
            * (
                d**2
                * m
                * rr[0]
                * (
                    d**2
                    * (m - 1)
                    * jnp.imag(
                        (-R + rr[0] + 1j * rr[2]) ** (m - 2) * jnp.exp(1j * n * rr[1])
                    )
                    + rr[2]
                    * jnp.real(
                        (-R + rr[0] + 1j * rr[2]) ** (m - 1) * jnp.exp(1j * n * rr[1])
                    )
                )
                - d**2
                * (
                    d**2
                    * m
                    * jnp.imag(
                        (-R + rr[0] + 1j * rr[2]) ** (m - 1) * jnp.exp(1j * n * rr[1])
                    )
                    + rr[2]
                    * jnp.real((-R + rr[0] + 1j * rr[2]) ** m * jnp.exp(1j * n * rr[1]))
                )
                + rr[0]
                * (R - rr[0])
                * (
                    d**2
                    * m
                    * jnp.imag(
                        (-R + rr[0] + 1j * rr[2]) ** (m - 1) * jnp.exp(1j * n * rr[1])
                    )
                    + rr[2]
                    * jnp.real((-R + rr[0] + 1j * rr[2]) ** m * jnp.exp(1j * n * rr[1]))
                )
            )
            * jnp.exp(
                (-(R**2) + 2 * R * rr[0] - rr[0] ** 2 - rr[2] ** 2) / (2 * d**2)
            )
            / (2 * jnp.sqrt(jnp.pi) * d**5 * rr[0] ** 2),
            jnp.sqrt(2)
            * n
            * (
                d**2
                * m
                * jnp.real((-R + rr[0] + 1j * rr[2]) ** (m - 1) * jnp.exp(1j * n * rr[1]))
                - rr[2]
                * jnp.imag((-R + rr[0] + 1j * rr[2]) ** m * jnp.exp(1j * n * rr[1]))
            )
            * jnp.exp(
                (-(R**2) + 2 * R * rr[0] - rr[0] ** 2 - rr[2] ** 2) / (2 * d**2)
            )
            / (2 * jnp.sqrt(jnp.pi) * d**3 * rr[0] ** 2),
            jnp.sqrt(2)
            * (
                d**2
                * (
                    d**2
                    * m
                    * (m - 1)
                    * jnp.real(
                        (-R + rr[0] + 1j * rr[2]) ** (m - 2) * jnp.exp(1j * n * rr[1])
                    )
                    - m
                    * rr[2]
                    * jnp.imag(
                        (-R + rr[0] + 1j * rr[2]) ** (m - 1) * jnp.exp(1j * n * rr[1])
                    )
                    + jnp.real((-R + rr[0] + 1j * rr[2]) ** m * jnp.exp(1j * n * rr[1]))
                )
                - rr[2]
                * (
                    d**2
                    * m
                    * jnp.imag(
                        (-R + rr[0] + 1j * rr[2]) ** (m - 1) * jnp.exp(1j * n * rr[1])
                    )
                    + rr[2]
                    * jnp.real((-R + rr[0] + 1j * rr[2]) ** m * jnp.exp(1j * n * rr[1]))
                )
            )
            * jnp.exp(
                (-(R**2) + 2 * R * rr[0] - rr[0] ** 2 - rr[2] ** 2) / (2 * d**2)
            )
            / (2 * jnp.sqrt(jnp.pi) * d**5 * rr[0]),
            0,
            0,
            0,
            jnp.sqrt(2)
            * (
                d**2
                * rr[0]
                * (
                    d**2
                    * m
                    * (m - 1)
                    * jnp.real(
                        (-R + rr[0] + 1j * rr[2]) ** (m - 2) * jnp.exp(1j * n * rr[1])
                    )
                    + m
                    * (R - rr[0])
                    * jnp.real(
                        (-R + rr[0] + 1j * rr[2]) ** (m - 1) * jnp.exp(1j * n * rr[1])
                    )
                    - jnp.real((-R + rr[0] + 1j * rr[2]) ** m * jnp.exp(1j * n * rr[1]))
                )
                - d**2
                * (
                    d**2
                    * m
                    * jnp.real(
                        (-R + rr[0] + 1j * rr[2]) ** (m - 1) * jnp.exp(1j * n * rr[1])
                    )
                    + (R - rr[0])
                    * jnp.real((-R + rr[0] + 1j * rr[2]) ** m * jnp.exp(1j * n * rr[1]))
                )
                + rr[0]
                * (R - rr[0])
                * (
                    d**2
                    * m
                    * jnp.real(
                        (-R + rr[0] + 1j * rr[2]) ** (m - 1) * jnp.exp(1j * n * rr[1])
                    )
                    + (R - rr[0])
                    * jnp.real((-R + rr[0] + 1j * rr[2]) ** m * jnp.exp(1j * n * rr[1]))
                )
            )
            * jnp.exp(
                (-(R**2) + 2 * R * rr[0] - rr[0] ** 2 - rr[2] ** 2) / (2 * d**2)
            )
            / (2 * jnp.sqrt(jnp.pi) * d**5 * rr[0] ** 2),
            jnp.sqrt(2)
            * n
            * (
                -(d**2)
                * m
                * jnp.imag((-R + rr[0] + 1j * rr[2]) ** (m - 1) * jnp.exp(1j * n * rr[1]))
                - (R - rr[0])
                * jnp.imag((-R + rr[0] + 1j * rr[2]) ** m * jnp.exp(1j * n * rr[1]))
            )
            * jnp.exp(
                (-(R**2) + 2 * R * rr[0] - rr[0] ** 2 - rr[2] ** 2) / (2 * d**2)
            )
            / (2 * jnp.sqrt(jnp.pi) * d**3 * rr[0] ** 2),
            jnp.sqrt(2)
            * (
                -(d**2)
                * m
                * (
                    d**2
                    * (m - 1)
                    * jnp.imag(
                        (-R + rr[0] + 1j * rr[2]) ** (m - 2) * jnp.exp(1j * n * rr[1])
                    )
                    + (R - rr[0])
                    * jnp.imag(
                        (-R + rr[0] + 1j * rr[2]) ** (m - 1) * jnp.exp(1j * n * rr[1])
                    )
                )
                - rr[2]
                * (
                    d**2
                    * m
                    * jnp.real(
                        (-R + rr[0] + 1j * rr[2]) ** (m - 1) * jnp.exp(1j * n * rr[1])
                    )
                    + (R - rr[0])
                    * jnp.real((-R + rr[0] + 1j * rr[2]) ** m * jnp.exp(1j * n * rr[1]))
                )
            )
            * jnp.exp(
                (-(R**2) + 2 * R * rr[0] - rr[0] ** 2 - rr[2] ** 2) / (2 * d**2)
            )
            / (2 * jnp.sqrt(jnp.pi) * d**5 * rr[0]),
        ]
    )
