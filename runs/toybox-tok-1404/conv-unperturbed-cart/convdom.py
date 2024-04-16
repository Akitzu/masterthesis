from pyoculus.problems import AnalyticCylindricalBfield
from horus.convergence_domain import convergence_domain, plot_convergence_domain
import matplotlib.pyplot as plt
import datetime
import numpy as np
import pickle

from pyoculus.problems import CylindricalBfield, CartesianBfield
import functools
import copy

def cartesianize_B(B):
    @functools.wraps(B)
    def wrapper(self, xyz, *args, **kwargs):
        r = np.linalg.norm(xyz[0:2])
        phi = np.arctan2(xyz[1], xyz[0])
        invjac = CartesianBfield._inv_Jacobian(r, phi, xyz[2])
        # return np.linalg.inv(jac) @ B([r, phi, xyz[2]], *args, **kwargs)
        return np.linalg.solve(invjac, B([r, phi, xyz[2]], *args, **kwargs))
    return wrapper

def cartesianize_dBdX(dBdX):
    @functools.wraps(dBdX)
    def wrapper(self, xyz, *args, **kwargs):
        r = np.linalg.norm(xyz[0:2])
        phi = np.arctan2(xyz[1], xyz[0])
        rphiz = np.array([r, phi, xyz[2]])
        invjac = CartesianBfield._inv_Jacobian(*rphiz)
        jac = np.linalg.inv(invjac)
        b, dbdx = dBdX(rphiz, *args, **kwargs)
        b = b[0]
        chris = np.array([
            [0, -b[1]*r, 0 ],
            [b[1]/r, b[0]/r, 0],
            [0, 0, 0]
        ])
        b = (jac @ b.reshape(3,1)).flatten()
        return [b], jac @ (dbdx + chris) @ invjac
    return wrapper

from types import MethodType

def tocartesian(cylindricalbfield):
    return_class = copy.deepcopy(cylindricalbfield)

    return_class.__class__ = CartesianBfield
    return_class.B = MethodType(cartesianize_B(copy.deepcopy(cylindricalbfield.B)), return_class)
    return_class.dBdX = MethodType(cartesianize_dBdX(copy.deepcopy(cylindricalbfield.dBdX)), return_class)

    def f_RZ_wrapper(self, *args, **kwargs):
        return CartesianBfield.f_RZ(self, *args, **kwargs)
    def f_RZ_tangent_wrapper(self, *args, **kwargs):
        return CartesianBfield.f_RZ_tangent(self, *args, **kwargs)
    
    return_class.f_RZ = MethodType(f_RZ_wrapper, return_class)
    return_class.f_RZ_tangent = MethodType(f_RZ_tangent_wrapper, return_class)

    return return_class

if __name__ == "__main__":
    ### Creating the pyoculus problem object
    print("\nCreating the pyoculus problem object\n")

    separatrix = {"type": "circular-current-loop", "amplitude": -4, "R": 3, "Z": -2.2}
    # Creating the pyoculus problem object, adding the perturbation here use the R, Z provided as center point
    pyoproblem_cyl = AnalyticCylindricalBfield.without_axis(
        3,
        0,
        0.91,
        0.7,
        perturbations_args=[separatrix],
        Rbegin=1,
        Rend=5,
        niter=800,
        guess=[3.0, -0.1],
        tol=1e-9,
    )
    pyoproblem = tocartesian(pyoproblem_cyl)

    # # Adding perturbation after the object is created uses the found axis as center point
    # pyoproblem.add_perturbation(maxwellboltzmann)

    ### Convergence domain calculation
    print("\nConvergence domain\n")

    rw = np.linspace(3.1025, 3.1125, 30)
    zw = np.linspace(-1.65, -1.66, 30)
    convdom = convergence_domain(
        pyoproblem,
        rw,
        zw,
        pp=0,
        qq=1,
        sbegin=2,
        send=6,
        tol=1e-10,
        checkonly=True,
        eps=1e-4,
        rtol=1e-10,
    )

    ### Plotting
    fig_perturbed = pickle.load(open("../poincare_04151810.pkl", "rb"))
    ax_perturbed = fig_perturbed.get_axes()[0]
    convdomplot = convdom[0:4]
    plot_convergence_domain(*convdomplot, ax_perturbed)
    ax_perturbed.legend()

    plt.show()

    date = datetime.datetime.now().strftime("%m%d%H%M")
    dumpname = f"convdom_{date}"    
    fig_perturbed.savefig(dumpname + ".png")
    with open(dumpname + ".pkl", "wb") as f:
        pickle.dump(fig_perturbed, f)
