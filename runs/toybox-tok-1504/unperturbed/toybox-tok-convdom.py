from pyoculus.problems import AnalyticCylindricalBfield
from horus.convergence_domain import convergence_domain, plot_convergence_domain
import matplotlib.pyplot as plt
import numpy as np
import datetime
import pickle

if __name__ == "__main__":
    ### Creating the pyoculus problem object
    print("\nCreating the pyoculus problem object\n")
    
    separatrix = {"type": "circular-current-loop", "amplitude": -10, "R": 6, "Z": -5.5}
    # gaussian10 = {"m": 1, "n": 0, "d": 1, "type": "gaussian", "amplitude": 0.01}

    # Creating the pyoculus problem object, adding the perturbation here use the R, Z provided as center point
    pyoproblem = AnalyticCylindricalBfield.without_axis(
        6,
        0,
        0.91,
        0.6,
        perturbations_args=[separatrix],
        Rbegin=1,
        Rend=8,
        niter=800,
        guess=[6.41, -0.7],
        tol=1e-10,
    )

    ### Convergence domain calculation
    rw = np.linspace(6.06, 6.33, 3)
    zw = np.linspace(-4.65, -4.369, 3)
    convdom = convergence_domain(
        pyoproblem,
        rw,
        zw,
        pp=0,
        qq=1,
        sbegin=2,
        send=8,
        tol=1e-10,
        checkonly=True,
        eps=1e-4,
        rtol=1e-10,
    )

    ### Plotting
    fig_perturbed = pickle.load(open("poincare_04160931.pkl", "rb"))
    ax_perturbed = fig_perturbed.get_axes()[0]
    convdomplot = convdom[0:4]
    plot_convergence_domain(*convdomplot, ax_perturbed)

    fig_perturbed.set_size_inches(10, 6)
    date = datetime.datetime.now().strftime("%m%d%H%M")
    dumpname = f"convergence_domain_{date}"
    with open(dumpname + ".pkl", "wb") as f:
        pickle.dump(fig_perturbed, f)

    plt.show()
    fig_perturbed.savefig(dumpname + ".png")

    breakpoint()