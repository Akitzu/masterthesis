import numpy as np
np.random.seed(0)
# pyoculus
from pyoculus.problems import CartesianBfield
from pyoculus.solvers import FixedPoint
# simsopt
from simsopt.configs import get_ncsx_data
from simsopt.field import (MagneticField, BiotSavart, coils_via_symmetries)

# Cartesian Magnetic field problem class for Simsopt Bfield 
class SimsoptBfieldProblem(CartesianBfield):
    def __init__(self, R0, Z0, Nfp, bs):
        super().__init__(R0, Z0, Nfp)

        if not isinstance(bs, MagneticField):
            raise ValueError("bs must be a MagneticField object")

        self.bs = bs

    # The return of the B field for the two following methods is not the same as the calls are :
    #   - CartesianBfield.f_RZ which does : 
    #   line 37     B = np.array([self.B(xyz, *args)]).T
    #   - CartesianBfield.f_RZ_tangent which does :  
    #   line 68     B, dBdX = self.dBdX(xyz, *args) 
    #   line 69     B = np.array(B).T
    # and both should result in a (3,1) array
    def B(self, xyz):
        xyz = np.reshape(xyz, (-1, 3))
        self.bs.set_points(xyz)
        return self.bs.B().flatten()

    def dBdX(self, xyz):
        B = self.B(xyz)
        return [B], self.bs.dB_by_dX().reshape(3, 3)

if __name__ == "__main__":
    # Set the number of field periods
    nfp = 3

    # Load the NCSX data and create the coils
    curves, currents, ma = get_ncsx_data()
    coils = coils_via_symmetries(curves, currents, nfp, True)

    # Create the Biot-Savart object
    bs = BiotSavart(coils)

    # ma.gamma()[0] = [x0,0.,0.] and initialize self.R0 self.Z0 as x0 and 0.
    ps = SimsoptBfieldProblem(ma.gamma()[0, 0], ma.gamma()[0, 1], nfp, bs)
    
    # Set up the integrator parameters
    iparams = dict()
    iparams["rtol"] = 1e-10

    # Set up the fixed point solver parameters    
    pparams = dict()
    pparams["nrestart"] = 100
    pparams["Z"] = 0

    fp01 = FixedPoint(ps, pparams, integrator_params=iparams)
    result01 = fp01.compute(guess=[1.5,0], pp=0, qq=1, sbegin=1, send=2)

    # Print of the results
    if fp01.is_successful():
        results = [list(p) for p in zip(result01.x, result01.y, result01.z)]
        print(results)
    else:
        print("Computation failed")