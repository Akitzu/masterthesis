#chaosfunctions.py: functions on euclidean space for field line integration.
#
#Integration routine defined in integrate.py
#
#Functions are derived using Mathematica in the notebook fluxfn_qprof_simlified.nb
#
#World's simplest toroidal flux function is used to generate a field with integrable
#surfaces on circular concentric tori with axis at R=1.
#
#A hermite-gaussian 1,1 perturbation function and others are also included,
#to investigate the transition into chaos when iota= .5+n.
#
#coded by Christophher Berg Smiet csmiet@pppl.gov
#File created 23-1-2019
import numpy as np
from numba import jit

@jit(nopython=True)
def torodal_field(xx, sf=2/3, shear=1):
    """
    Returns the magnetic field vector at position *xx* of a divergence free
    vector field whose integral curves lie on concentric circular tori with
    axis at $R=1$.
    the safety factor profile goes as (sf + shear*a^2)/sqrt(1-a^2)
    *xx*:
        position where the field is to be evaluated
    *sf*:
        safety factor on the axis
    *shear*:
        multiplication factor to increase the steepness of the q-profile. Not implemented!
    """
    B_x =   (2*(-(xx[0]*xx[2]) + xx[1]*np.sqrt(-xx[0]**2 - xx[1]**2 + 2*np.sqrt(xx[0]**2 + xx[1]**2) - xx[2]**2)* \
            (sf + (-1 + np.sqrt(xx[0]**2 + xx[1]**2))**2 + xx[2]**2)))/(xx[0]**2 + xx[1]**2)
    B_y =   (2*(-(xx[1]*xx[2]) - xx[0]*np.sqrt(-xx[0]**2 - xx[1]**2 + 2*np.sqrt(xx[0]**2 + xx[1]**2) - xx[2]**2)* \
            (sf + (-1 + np.sqrt(xx[0]**2 + xx[1]**2))**2 + xx[2]**2)))/(xx[0]**2 + xx[1]**2)
    B_z =   2 - 2/np.sqrt(xx[0]**2 + xx[1]**2)
    return np.array([B_x, B_y, B_z])

@jit(nopython=True)
def torodal_field_squared_q(xx, sf=2/3, shear=1, epsilon = 0):
    """
    Returns the magnetic field vector at position *xx* of a divergence free
    vector field whose integral curves lie on concentric circular tori with
    axis at $R=1$.
    the safety factor profile goes as (sf + shear*a^2)
    *xx*:
        position where the field is to be evaluated
    *sf*:
        safety factor on the axis
    *shear*:
        multiplication factor to increase the steepness of the q-profile. Not implemented!
    """
    if epsilon == 1: return np.array((0.,0.,0.))
    B = np.zeros(3)

    B_x = (2*(xx[0]*xx[2] - xx[1]*np.sqrt(-xx[0]**2 - xx[1]**2 + 2*np.sqrt(xx[0]**2 + xx[1]**2) - xx[2]**2)*(sf + (-1 + np.sqrt(xx[0]**2 + xx[1]**2))**2 + xx[2]**2)))/(xx[0]**2 + xx[1]**2)
    B_y = (2*(xx[1]*xx[2] + xx[0]*np.sqrt(-xx[0]**2 - xx[1]**2 + 2*np.sqrt(xx[0]**2 + xx[1]**2) - xx[2]**2)*(sf + (-1 + np.sqrt(xx[0]**2 + xx[1]**2))**2 + xx[2]**2)))/(xx[0]**2 + xx[1]**2)
    B_z = -2 + 2/np.sqrt(xx[0]**2 + xx[1]**2)
    return (1-epsilon) * np.array([B_x, B_y, B_z])


@jit(nopython=True)
def torodal_field_old(xx, sf=2/3, shear=1):
    """
    Returns the magnetic field vector at position *xx* of a divergence free
    vector field whose integral curves lie on concentric circular tori with
    axis at $R=1$.
    the safety factor profile goes as q_0/sqrt(1-a^2)
    Arguments:
    *xx*:
        position where the field is to be evaluated
    *sf*:
        safety factor on the axis
    *shear*:
        multiplication factor to increase the steepness of the q-profile. Not implemented!
    """
    B_x = (2*(-2*xx[1] + 3*xx[0]*xx[2]))/(3.*(xx[0]**2 + xx[1]**2))
    B_y = (2*(2*xx[0] + 3*xx[1]*xx[2]))/(3.*(xx[0]**2 + xx[1]**2))
    B_z = -2 + 2/np.sqrt(xx[0]**2 + xx[1]**2)
    return np.array([B_x, B_y, B_z])


@jit(nopython=True)
def perturb_23(xx, width = 0.1, amplitude =1):
    """
    Poloidal number first (exponent in the poloidal direction), then toroidal number.
    """
    if amplitude == 0: return np.array((0.,0.,0.))
    B = np.zeros(3)

    B[0] = (xx[0]*(2*np.sin(3*np.arctan2(xx[0],xx[1]))*(-1 + np.sqrt(xx[0]**2 + xx[1]**2))*(-width + xx[2])*(width + xx[2]) + \
           np.cos(3*np.arctan2(xx[0],xx[1]))*xx[2]*(2*width**2 + (-1 + np.sqrt(xx[0]**2 + xx[1]**2))**2 - xx[2]**2)))/\
           (np.exp(((-1 + np.sqrt(xx[0]**2 + xx[1]**2))**2 + xx[2]**2)/(2.*width**2))*width**2*(xx[0]**2 + xx[1]**2))

    B[1] = (xx[1]*(2*np.sin(3*np.arctan2(xx[0],xx[1]))*(-1 + np.sqrt(xx[0]**2 + xx[1]**2))*(-width + xx[2])*(width + xx[2]) +\
            np.cos(3*np.arctan2(xx[0],xx[1]))*xx[2]*(2*width**2 + (-1 + np.sqrt(xx[0]**2 + xx[1]**2))**2 - xx[2]**2)))/\
           (np.exp(((-1 + np.sqrt(xx[0]**2 + xx[1]**2))**2 + xx[2]**2)/(2.*width**2))*width**2*(xx[0]**2 + xx[1]**2))

    B[2] = (2*np.sin(3*np.arctan2(xx[0],xx[1]))*(-1 + width**2 - xx[0]**2 - xx[1]**2 + 2*np.sqrt(xx[0]**2 + xx[1]**2))*xx[2] + \
           np.cos(3*np.arctan2(xx[0],xx[1]))*(-1 + np.sqrt(xx[0]**2 + xx[1]**2))*(2*width**2 - (-1 + np.sqrt(xx[0]**2 + xx[1]**2))**2 + xx[2]**2))/\
           (np.exp(((-1 + np.sqrt(xx[0]**2 + xx[1]**2))**2 + xx[2]**2)/(2.*width**2))*width**2*np.sqrt(xx[0]**2 + xx[1]**2))

    return B*amplitude


@jit(nopython=True)
def perturb_12(xx, width = 0.1, amplitude =1):
    """
    """
    if amplitude == 0: return np.array((0.,0.,0.))
    B = np.zeros(3)

    B[0] = (xx[0]*(np.cos(2*np.arctan2(xx[0],xx[1]))*(-1 + np.sqrt(xx[0]**2 + xx[1]**2))*xx[2] + np.sin(2*np.arctan2(xx[0],xx[1]))*(-width + xx[2])*(width + xx[2])))/ \
           (np.exp(((-1 + np.sqrt(xx[0]**2 + xx[1]**2))**2 + xx[2]**2)/(2.*width**2))*width**2*(xx[0]**2 + xx[1]**2))

    B[1] = (xx[1]*(np.cos(2*np.arctan2(xx[0],xx[1]))*(-1 + np.sqrt(xx[0]**2 + xx[1]**2))*xx[2] + np.sin(2*np.arctan2(xx[0],xx[1]))*(-width + xx[2])*(width + xx[2])))/\
           (np.exp(((-1 + np.sqrt(xx[0]**2 + xx[1]**2))**2 + xx[2]**2)/(2.*width**2))*width**2*(xx[0]**2 + xx[1]**2))

    B[2] = (np.cos(2*np.arctan2(xx[0],xx[1]))*(-1 + width**2 - xx[0]**2 - xx[1]**2 + 2*np.sqrt(xx[0]**2 + xx[1]**2)) - np.sin(2*np.arctan2(xx[0],xx[1]))*(-1 + np.sqrt(xx[0]**2 + xx[1]**2))*xx[2])/\
           (np.exp(((-1 + np.sqrt(xx[0]**2 + xx[1]**2))**2 + xx[2]**2)/(2.*width**2))*width**2*np.sqrt(xx[0]**2 + xx[1]**2))

    return B*amplitude


@jit(nopython=True)
def perturb_21(xx, width = 0.1, amplitude =1):
    """
    """
    if amplitude == 0: return np.array((0.,0.,0.))
    B = np.zeros(3)

    B[0] = (xx[0]*(2*xx[1]*(-1 + np.sqrt(xx[0]**2 + xx[1]**2))*(-width + xx[2])*(width + xx[2]) + xx[0]*xx[2]*(2*width**2 + (-1 + np.sqrt(xx[0]**2 + xx[1]**2))**2 - xx[2]**2)))/\
           (np.exp(((-1 + np.sqrt(xx[0]**2 + xx[1]**2))**2 + xx[2]**2)/(2.*width**2))*width**2*(xx[0]**2 + xx[1]**2)**1.5)

    B[1] = (xx[1]*(2*xx[1]*(-1 + np.sqrt(xx[0]**2 + xx[1]**2))*(-width + xx[2])*(width + xx[2]) + xx[0]*xx[2]*(2*width**2 + (-1 + np.sqrt(xx[0]**2 + xx[1]**2))**2 - xx[2]**2)))/\
           (np.exp(((-1 + np.sqrt(xx[0]**2 + xx[1]**2))**2 + xx[2]**2)/(2.*width**2))*width**2*(xx[0]**2 + xx[1]**2)**1.5)

    B[2] = (-2*xx[1]*(1 - width**2 + xx[0]**2 + xx[1]**2 - 2*np.sqrt(xx[0]**2 + xx[1]**2))*xx[2] + \
            xx[0]*(-1 + np.sqrt(xx[0]**2 + xx[1]**2))*(2*width**2 - (-1 + np.sqrt(xx[0]**2 + xx[1]**2))**2 + xx[2]**2))/\
           (np.exp(((-1 + np.sqrt(xx[0]**2 + xx[1]**2))**2 + xx[2]**2)/(2.*width**2))*width**2*(xx[0]**2 + xx[1]**2))

    return B*amplitude


@jit(nopython=True)
def perturb_11_second(xx, width = 0.1, amplitude =1):
    if amplitude == 0: return np.array((0.,0.,0.))
    B = np.zeros(3)

    B[0] = (xx[0]*(-(width**2*xx[1]) + xx[2]*(xx[0]*(-1 + np.sqrt(xx[0]**2 + xx[1]**2)) + xx[1]*xx[2])))/ \
           (np.exp((1 + xx[0]**2 + xx[1]**2 - 2*np.sqrt(xx[0]**2 + xx[1]**2) + xx[2]**2)/(2.*width**2))*width**2*(xx[0]**2 + xx[1]**2)**1.5)

    B[1] = (xx[1]*(-(width**2*xx[1]) + xx[2]*(xx[0]*(-1 + np.sqrt(xx[0]**2 + xx[1]**2)) + xx[1]*xx[2])))/ \
           (np.exp((1 + xx[0]**2 + xx[1]**2 - 2*np.sqrt(xx[0]**2 + xx[1]**2) + xx[2]**2)/(2.*width**2))*width**2*(xx[0]**2 + xx[1]**2)**1.5)

    B[2] = (-xx[0]**3 + xx[0]*(-1 + width**2 - xx[1]**2 + 2*np.sqrt(xx[0]**2 + xx[1]**2)) - xx[1]*(-1 + np.sqrt(xx[0]**2 + xx[1]**2))*xx[2])/ \
           (np.exp((1 + xx[0]**2 + xx[1]**2 - 2*np.sqrt(xx[0]**2 + xx[1]**2) + xx[2]**2)/(2.*width**2))*width**2*(xx[0]**2 + xx[1]**2))

    return B*amplitude



@jit(nopython=True)
def perturb_11(xx, width = 0.1, amplitude =1):
    """
    This perturbation function rotates around the axis 1.5 times, as it should. It has poloidal mode number 1.
    """
    if amplitude == 0: return np.array((0.,0.,0.))
    B = np.zeros(3)

    B[0] = (xx[0]*(-(width**2*xx[1]) + xx[2]*(xx[0]*(-1 + np.sqrt(xx[0]**2 + xx[1]**2)) + xx[1]*xx[2])))/ \
           (np.exp((1 + xx[0]**2 + xx[1]**2 - 2*np.sqrt(xx[0]**2 + xx[1]**2) + xx[2]**2)/(2.*width**2))*width**2*(xx[0]**2 + xx[1]**2)**1.5)

    B[1] = (xx[1]*(-(width**2*xx[1]) + xx[2]*(xx[0]*(-1 + np.sqrt(xx[0]**2 + xx[1]**2)) + xx[1]*xx[2])))/ \
           (np.exp((1 + xx[0]**2 + xx[1]**2 - 2*np.sqrt(xx[0]**2 + xx[1]**2) + xx[2]**2)/(2.*width**2))*width**2*(xx[0]**2 + xx[1]**2)**1.5)

    B[2] = (-xx[0]**3 + xx[0]*(-1 + width**2 - xx[1]**2 + 2*np.sqrt(xx[0]**2 + xx[1]**2)) - xx[1]*(-1 + np.sqrt(xx[0]**2 + xx[1]**2))*xx[2])/ \
           (np.exp((1 + xx[0]**2 + xx[1]**2 - 2*np.sqrt(xx[0]**2 + xx[1]**2) + xx[2]**2)/(2.*width**2))*width**2*(xx[0]**2 + xx[1]**2))

    return B*amplitude


@jit(nopython=True)
def perturb_31(xx, width = 0.1, amplitude =1):
    """
    """
    if amplitude == 0: return np.array((0.,0.,0.))
    B = np.zeros(3)

    B[0] = (xx[0]*(xx[0]*(-1 + np.sqrt(xx[0]**2 + xx[1]**2))*xx[2]*(6*width**2 + (-1 + np.sqrt(xx[0]**2 + xx[1]**2))**2 - 3*xx[2]**2) - \
           xx[1]*(3*width**2*(-1 + np.sqrt(xx[0]**2 + xx[1]**2))**2 - 3*(width**2 + (-1 + np.sqrt(xx[0]**2 + xx[1]**2))**2)*xx[2]**2 + xx[2]**4)))/\
           (np.exp(((-1 + np.sqrt(xx[0]**2 + xx[1]**2))**2 + xx[2]**2)/(2.*width**2))*width**2*(xx[0]**2 + xx[1]**2)**1.5)

    B[1] = (xx[1]*(xx[0]*(-1 + np.sqrt(xx[0]**2 + xx[1]**2))*xx[2]*(6*width**2 + (-1 + np.sqrt(xx[0]**2 + xx[1]**2))**2 - 3*xx[2]**2) - \
           xx[1]*(3*width**2*(-1 + np.sqrt(xx[0]**2 + xx[1]**2))**2 - 3*(width**2 + (-1 + np.sqrt(xx[0]**2 + xx[1]**2))**2)*xx[2]**2 + xx[2]**4)))/\
           (np.exp(((-1 + np.sqrt(xx[0]**2 + xx[1]**2))**2 + xx[2]**2)/(2.*width**2))*width**2*(xx[0]**2 + xx[1]**2)**1.5)

    B[2] = (xx[1]*(-1 + np.sqrt(xx[0]**2 + xx[1]**2))*xx[2]*(6*width**2 - 3*(-1 + np.sqrt(xx[0]**2 + xx[1]**2))**2 + xx[2]**2) + \
           xx[0]*(-(-1 + np.sqrt(xx[0]**2 + xx[1]**2))**4 + 3*(-1 + np.sqrt(xx[0]**2 + xx[1]**2))**2*xx[2]**2 + 3*width**2*((-1 + np.sqrt(xx[0]**2 + xx[1]**2))**2 - xx[2]**2)))/ \
          (np.exp(((-1 + np.sqrt(xx[0]**2 + xx[1]**2))**2 + xx[2]**2)/(2.*width**2))*width**2*(xx[0]**2 + xx[1]**2))
    return B*amplitude


@jit(nopython=True)
def perturb_33(xx, width = 0.1, amplitude =1):
    """
    """
    if amplitude == 0: return np.array((0.,0.,0.))
    B = np.zeros(3)

    B[0] = (xx[0]*(np.cos(3*np.arctan2(xx[0],xx[1]))*(-1 + np.sqrt(xx[0]**2 + xx[1]**2))*xx[2]*(6*width**2 + (-1 + np.sqrt(xx[0]**2 + xx[1]**2))**2 - 3*xx[2]**2) - \
           np.sin(3*np.arctan2(xx[0],xx[1]))*(3*width**2*(-1 + np.sqrt(xx[0]**2 + xx[1]**2))**2 - 3*(width**2 + (-1 + np.sqrt(xx[0]**2 + xx[1]**2))**2)*xx[2]**2 + xx[2]**4)))/\
           (np.exp(((-1 + np.sqrt(xx[0]**2 + xx[1]**2))**2 + xx[2]**2)/(2.*width**2))*width**2*(xx[0]**2 + xx[1]**2))

    B[1] = (xx[1]*(np.cos(3*np.arctan2(xx[0],xx[1]))*(-1 + np.sqrt(xx[0]**2 + xx[1]**2))*xx[2]*(6*width**2 + (-1 + np.sqrt(xx[0]**2 + xx[1]**2))**2 - 3*xx[2]**2) - \
           np.sin(3*np.arctan2(xx[0],xx[1]))*(3*width**2*(-1 + np.sqrt(xx[0]**2 + xx[1]**2))**2 - 3*(width**2 + (-1 + np.sqrt(xx[0]**2 + xx[1]**2))**2)*xx[2]**2 + xx[2]**4)))/\
          (np.exp(((-1 + np.sqrt(xx[0]**2 + xx[1]**2))**2 + xx[2]**2)/(2.*width**2))*width**2*(xx[0]**2 + xx[1]**2))

    B[2] = (np.sin(3*np.arctan2(xx[0],xx[1]))*(-1 + np.sqrt(xx[0]**2 + xx[1]**2))*xx[2]*(6*width**2 - 3*(-1 + np.sqrt(xx[0]**2 + xx[1]**2))**2 + xx[2]**2) + \
           np.cos(3*np.arctan2(xx[0],xx[1]))*(-(-1 + np.sqrt(xx[0]**2 + xx[1]**2))**4 + 3*(-1 + np.sqrt(xx[0]**2 + xx[1]**2))**2*xx[2]**2 + 3*width**2*((-1 + np.sqrt(xx[0]**2 + xx[1]**2))**2 - xx[2]**2)))/ \
          (np.exp(((-1 + np.sqrt(xx[0]**2 + xx[1]**2))**2 + xx[2]**2)/(2.*width**2))*width**2*np.sqrt(xx[0]**2 + xx[1]**2))

    return B*amplitude


@jit(nopython=True)
def perturb_22(xx, width = 0.1, amplitude =1):
    """
    """
    if amplitude == 0: return np.array((0.,0.,0.))
    B = np.zeros(3)

    B[0] = (xx[0]*(2*np.sin(2*np.arctan2(xx[0],xx[1]))*(-1 + np.sqrt(xx[0]**2 + xx[1]**2))*(-width + xx[2])*(width + xx[2]) + \
           np.cos(2*np.arctan2(xx[0],xx[1]))*xx[2]*(2*width**2 + (-1 + np.sqrt(xx[0]**2 + xx[1]**2))**2 - xx[2]**2)))/\
           (np.exp(((-1 + np.sqrt(xx[0]**2 + xx[1]**2))**2 + xx[2]**2)/(2.*width**2))*width**2*(xx[0]**2 + xx[1]**2))

    B[1] = (xx[1]*(2*np.sin(2*np.arctan2(xx[0],xx[1]))*(-1 + np.sqrt(xx[0]**2 + xx[1]**2))*(-width + xx[2])*(width + xx[2]) + \
            np.cos(2*np.arctan2(xx[0],xx[1]))*xx[2]*(2*width**2 + (-1 + np.sqrt(xx[0]**2 + xx[1]**2))**2 - xx[2]**2)))/ \
           (np.exp(((-1 + np.sqrt(xx[0]**2 + xx[1]**2))**2 + xx[2]**2)/(2.*width**2))*width**2*(xx[0]**2 + xx[1]**2))

    B[2] = (2*np.sin(2*np.arctan2(xx[0],xx[1]))*(-1 + width**2 - xx[0]**2 - xx[1]**2 + 2*np.sqrt(xx[0]**2 + xx[1]**2))*xx[2] + \
           np.cos(2*np.arctan2(xx[0],xx[1]))*(-1 + np.sqrt(xx[0]**2 + xx[1]**2))*(2*width**2 - (-1 + np.sqrt(xx[0]**2 + xx[1]**2))**2 + xx[2]**2))/ \
          (np.exp(((-1 + np.sqrt(xx[0]**2 + xx[1]**2))**2 + xx[2]**2)/(2.*width**2))*width**2*np.sqrt(xx[0]**2 + xx[1]**2))

    return B*amplitude


@jit(nopython=True)
def perturb_32(xx, width = 0.1, amplitude =1):
    """
    """
    if amplitude == 0: return np.array((0.,0.,0.))
    B = np.zeros(3)

    B[0] = (xx[0]*(2*np.sin(2*np.arctan2(xx[0],xx[1]))*(-1 + np.sqrt(xx[0]**2 + xx[1]**2))*(-width + xx[2])*(width + xx[2]) + \
           np.cos(2*np.arctan2(xx[0],xx[1]))*xx[2]*(2*width**2 + (-1 + np.sqrt(xx[0]**2 + xx[1]**2))**2 - xx[2]**2)))/\
           (np.exp(((-1 + np.sqrt(xx[0]**2 + xx[1]**2))**2 + xx[2]**2)/(2.*width**2))*width**2*(xx[0]**2 + xx[1]**2))

    B[1] = (xx[1]*(2*np.sin(2*np.arctan2(xx[0],xx[1]))*(-1 + np.sqrt(xx[0]**2 + xx[1]**2))*(-width + xx[2])*(width + xx[2]) + \
            np.cos(2*np.arctan2(xx[0],xx[1]))*xx[2]*(2*width**2 + (-1 + np.sqrt(xx[0]**2 + xx[1]**2))**2 - xx[2]**2)))/ \
           (np.exp(((-1 + np.sqrt(xx[0]**2 + xx[1]**2))**2 + xx[2]**2)/(2.*width**2))*width**2*(xx[0]**2 + xx[1]**2))

    B[2] = (2*np.sin(2*np.arctan2(xx[0],xx[1]))*(-1 + width**2 - xx[0]**2 - xx[1]**2 + 2*np.sqrt(xx[0]**2 + xx[1]**2))*xx[2] + \
           np.cos(2*np.arctan2(xx[0],xx[1]))*(-1 + np.sqrt(xx[0]**2 + xx[1]**2))*(2*width**2 - (-1 + np.sqrt(xx[0]**2 + xx[1]**2))**2 + xx[2]**2))/ \
          (np.exp(((-1 + np.sqrt(xx[0]**2 + xx[1]**2))**2 + xx[2]**2)/(2.*width**2))*width**2*np.sqrt(xx[0]**2 + xx[1]**2))

    return B*amplitude


@jit(nopython=True)
def perturb_kink(xx, width=0.2, amplitude=1, sf=0.9, shear=2.5):
    """
    Analytical function for the perturbed field corresponding to a kink displacement.
    The kink displacement has gaussian amplitude localized on R=1 with characteristic
    width (width).
    Given by B_1=Curl[Cross[xi, B_0]] where xi = (1/R dpsi/dz, 0, 1/R
    Spare your sanity and do not look into the monster expression that mathematica has created.
    Even this monster has Div[B]<10e-12.


    """

    if amplitude == 0: return np.array((0.,0.,0.))
    B = np.zeros(3)

    B[0] = (2*(xx[0]*(-(width**2*xx[0]) + xx[2]*(xx[1] - xx[1]*np.sqrt(xx[0]**2 + xx[1]**2) + xx[0]*xx[2]))*\
            (np.sqrt(xx[0]**2 + xx[1]**2) + np.sqrt(-xx[0]**2 - xx[1]**2 + 2*np.sqrt(xx[0]**2 + xx[1]**2) - xx[2]**2)*(sf + shear*(1 + xx[0]**2 + xx[1]**2 - 2*np.sqrt(xx[0]**2 + xx[1]**2) + xx[2]**2))) - \
           (xx[1]*(xx[0]*xx[2]*(sf*np.sqrt(xx[0]**2 + xx[1]**2)*(-4 + width**2 - 2*xx[0]**2 - 2*xx[1]**2 + 6*np.sqrt(xx[0]**2 + xx[1]**2)) + \
                   shear*np.sqrt(xx[0]**2 + xx[1]**2)*(width**2*(1 + 3*xx[0]**2 + 3*xx[1]**2 - 6*np.sqrt(xx[0]**2 + xx[1]**2)) - 2*(-2 + np.sqrt(xx[0]**2 + xx[1]**2))*(-1 + np.sqrt(xx[0]**2 + xx[1]**2))**3) + \
                   (-(shear*np.sqrt(xx[0]**2 + xx[1]**2)*(10 - 3*width**2 + 4*xx[0]**2 + 4*xx[1]**2)) + 2*shear*(1 + 6*xx[0]**2 + 6*xx[1]**2) - 2*sf*(-1 + np.sqrt(xx[0]**2 + xx[1]**2)))*xx[2]**2 - \
                   2*shear*(-1 + np.sqrt(xx[0]**2 + xx[1]**2))*xx[2]**4) - xx[1]*(width**2*\
                    (np.sqrt(xx[0]**2 + xx[1]**2)*(-(sf*(-3 + np.sqrt(xx[0]**2 + xx[1]**2))) + shear*(-3 + xx[0]**2 + xx[1]**2)*(-1 + np.sqrt(xx[0]**2 + xx[1]**2))) - \
                      (2*sf + shear*(2 + xx[0]**2 + xx[1]**2 - 5*np.sqrt(xx[0]**2 + xx[1]**2)))*xx[2]**2 - 2*shear*xx[2]**4) + \
                   2*xx[2]**2*(xx[0]**2 + xx[1]**2 - 2*np.sqrt(xx[0]**2 + xx[1]**2) + xx[2]**2)*(sf + shear*((-1 + np.sqrt(xx[0]**2 + xx[1]**2))**2 + xx[2]**2)))))/\
            np.sqrt(-xx[0]**2 - xx[1]**2 + 2*np.sqrt(xx[0]**2 + xx[1]**2) - xx[2]**2)))/(np.exp(((-1 + np.sqrt(xx[0]**2 + xx[1]**2))**2 + xx[2]**2)/(2.*width**2))*width**2*(xx[0]**2 + xx[1]**2)**2.5)

    B[1] = (2*(-(xx[1]*(xx[1]*(-1 + np.sqrt(xx[0]**2 + xx[1]**2))*xx[2] + xx[0]*(width - xx[2])*(width + xx[2]))*\
             (np.sqrt(xx[0]**2 + xx[1]**2) + np.sqrt(-xx[0]**2 - xx[1]**2 + 2*np.sqrt(xx[0]**2 + xx[1]**2) - xx[2]**2)*(sf + shear*(1 + xx[0]**2 + xx[1]**2 - 2*np.sqrt(xx[0]**2 + xx[1]**2) + xx[2]**2)))) + \
           (xx[0]*(xx[0]*xx[2]*(sf*np.sqrt(xx[0]**2 + xx[1]**2)*(-4 + width**2 - 2*xx[0]**2 - 2*xx[1]**2 + 6*np.sqrt(xx[0]**2 + xx[1]**2)) + \
                  shear*np.sqrt(xx[0]**2 + xx[1]**2)*(width**2*(1 + 3*xx[0]**2 + 3*xx[1]**2 - 6*np.sqrt(xx[0]**2 + xx[1]**2)) - 2*(-2 + np.sqrt(xx[0]**2 + xx[1]**2))*(-1 + np.sqrt(xx[0]**2 + xx[1]**2))**3) + \
                  (-(shear*np.sqrt(xx[0]**2 + xx[1]**2)*(10 - 3*width**2 + 4*xx[0]**2 + 4*xx[1]**2)) + 2*shear*(1 + 6*xx[0]**2 + 6*xx[1]**2) - 2*sf*(-1 + np.sqrt(xx[0]**2 + xx[1]**2)))*xx[2]**2 - \
                  2*shear*(-1 + np.sqrt(xx[0]**2 + xx[1]**2))*xx[2]**4) - xx[1]*(width**2*\
                   (np.sqrt(xx[0]**2 + xx[1]**2)*(-(sf*(-3 + np.sqrt(xx[0]**2 + xx[1]**2))) + shear*(-3 + xx[0]**2 + xx[1]**2)*(-1 + np.sqrt(xx[0]**2 + xx[1]**2))) - \
                     (2*sf + shear*(2 + xx[0]**2 + xx[1]**2 - 5*np.sqrt(xx[0]**2 + xx[1]**2)))*xx[2]**2 - 2*shear*xx[2]**4) + \
                  2*xx[2]**2*(xx[0]**2 + xx[1]**2 - 2*np.sqrt(xx[0]**2 + xx[1]**2) + xx[2]**2)*(sf + shear*((-1 + np.sqrt(xx[0]**2 + xx[1]**2))**2 + xx[2]**2)))))/\
           np.sqrt(-xx[0]**2 - xx[1]**2 + 2*np.sqrt(xx[0]**2 + xx[1]**2) - xx[2]**2)))/(np.exp(((-1 + np.sqrt(xx[0]**2 + xx[1]**2))**2 + xx[2]**2)/(2.*width**2))*width**2*(xx[0]**2 + xx[1]**2)**2.5)

    B[2] = (-2*(xx[0]*xx[2]*(width**2 + (-1 + np.sqrt(xx[0]**2 + xx[1]**2))*(np.sqrt(xx[0]**2 + xx[1]**2) + \
                 np.sqrt(-xx[0]**2 - xx[1]**2 + 2*np.sqrt(xx[0]**2 + xx[1]**2) - xx[2]**2)*(sf + shear*(1 + xx[0]**2 + xx[1]**2 - 2*np.sqrt(xx[0]**2 + xx[1]**2) + xx[2]**2)))) + \
           xx[1]*(-(np.sqrt(xx[0]**2 + xx[1]**2)*(-1 + np.sqrt(xx[0]**2 + xx[1]**2))**2) - sf*np.sqrt(-xx[0]**2 - xx[1]**2 + 2*np.sqrt(xx[0]**2 + xx[1]**2) - xx[2]**2) - \
              np.sqrt(-xx[0]**2 - xx[1]**2 + 2*np.sqrt(xx[0]**2 + xx[1]**2) - xx[2]**2)*(shear + \
                 np.sqrt(xx[0]**2 + xx[1]**2)*(-2 + np.sqrt(xx[0]**2 + xx[1]**2))*(sf + shear*(2 + xx[0]**2 + xx[1]**2 - 2*np.sqrt(xx[0]**2 + xx[1]**2))) + shear*(-1 + np.sqrt(xx[0]**2 + xx[1]**2))**2*xx[2]**2) + \
              width**2*(1 + np.sqrt(-xx[0]**2 - xx[1]**2 + 2*np.sqrt(xx[0]**2 + xx[1]**2) - xx[2]**2)*(sf + shear*((-1 + np.sqrt(xx[0]**2 + xx[1]**2))**2 + xx[2]**2))))))/\
           (np.exp(((-1 + np.sqrt(xx[0]**2 + xx[1]**2))**2 + xx[2]**2)/(2.*width**2))*width**2*(xx[0]**2 + xx[1]**2)**2)

    return B*amplitude





@jit(nopython=True)
def torodal_field_squared_q_final(xx, sf=2/3, shear=2.5, epsilon = 1.0):
    """
    Analytical function for the final state after Kadomstev reconnection
    The current jump normally present in these models is ignored in this
    case.
    """

    if epsilon == 0: return np.array((0.,0.,0.))
    B = np.zeros(3)

    B[0] =  (16*xx[1]*np.sqrt(-xx[0]**2 - xx[1]**2 + 2*np.sqrt(xx[0]**2 + xx[1]**2) - xx[2]**2)* \
            (sf + (-1 + np.sqrt(xx[0]**2 + xx[1]**2))**2 + xx[2]**2) - \
            4*xx[0]*xx[2]*(4*sf + 3*((-1 + np.sqrt(xx[0]**2 + xx[1]**2))**2 + xx[2]**2)))/ \
            (8.*(xx[0]**2 + xx[1]**2))

    B[1] =  -(16*xx[0]*np.sqrt(-xx[0]**2 - xx[1]**2 + 2*np.sqrt(xx[0]**2 + xx[1]**2) - xx[2]**2)* \
            (sf + (-1 + np.sqrt(xx[0]**2 + xx[1]**2))**2 + xx[2]**2) +  \
            4*xx[1]*xx[2]*(4*sf + 3*((-1 + np.sqrt(xx[0]**2 + xx[1]**2))**2 + xx[2]**2)))/ \
            (8.*(xx[0]**2 + xx[1]**2))

    B[2] =  -((-1 + np.sqrt(xx[0]**2 + xx[1]**2))* \
            (-4*sf - 3*xx[0]**2 + 6*np.sqrt(xx[0]**2 + xx[1]**2) - 3*(1 + xx[1]**2 + xx[2]**2)))/ \
            (2.*np.sqrt(xx[0]**2 + xx[1]**2))

    return -epsilon * B




@jit(nopython=True)
def ralf_pert(xx, width=0.2, amplitude = 1.0, ang=0.0):
    """
    (1,1) Kink as calculated in the notebook.
    """

    if amplitude == 0: return np.array((0.,0.,0.))
    B = np.zeros(3)

    B[0] =  (2*xx[0]**2*(-1 + np.sqrt(xx[0]**2 + xx[1]**2))*xx[2])/(np.exp(((-1 + np.sqrt(xx[0]**2 + xx[1]**2))**2 + xx[2]**2)/width**2)*width**2*(xx[0]**2 + xx[1]**2)**1.5)

    B[1] =  (2*xx[0]*xx[1]*(-1 + np.sqrt(xx[0]**2 + xx[1]**2))*xx[2])/(np.exp(((-1 + np.sqrt(xx[0]**2 + xx[1]**2))**2 + xx[2]**2)/width**2)*width**2*(xx[0]**2 + xx[1]**2)**1.5)

    B[2] =  (xx[0]*(-2 + width**2 - 2*xx[0]**2 - 2*xx[1]**2 + 4*np.sqrt(xx[0]**2 + xx[1]**2)))/ \
            (np.exp((1 + xx[0]**2 + xx[1]**2 - 2*np.sqrt(xx[0]**2 + xx[1]**2) + xx[2]**2)/width**2)*width**2*(xx[0]**2 + xx[1]**2))

    return amplitude * B
