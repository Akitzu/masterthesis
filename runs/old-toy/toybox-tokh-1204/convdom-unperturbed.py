from pyoculus.problems import AnalyticCylindricalBfield
from horus.convergence_domain import convergence_domain, plot_convergence_domain
import matplotlib.pyplot as plt
import datetime
import numpy as np
import pickle

if __name__ == "__main__":
    ### Creating the pyoculus problem object
    print("\nCreating the pyoculus problem object\n")

    separatrix = {"type": "circular-current-loop", "amplitude": -4, "R": 9, "Z": 0.0}

    # Creating the pyoculus problem object, adding the perturbation here use the R, Z provided as center point
    pyoproblem = AnalyticCylindricalBfield.without_axis(
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

    # # Adding perturbation after the object is created uses the found axis as center point
    # pyoproblem.add_perturbation(maxwellboltzmann)

    ### Convergence domain calculation
    rw = np.linspace(4.435, 4.438, 30)
    zw = np.linspace(-1.23, -1.22, 30)
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
    fig_perturbed = pickle.load(open("poincare_04151810.pkl", "rb"))
    ax_perturbed = fig_perturbed.get_axes()[0]
    convdomplot = convdom[0:4]
    plot_convergence_domain(*convdomplot, ax_perturbed)
    plt.show()

    date = datetime.datetime.now().strftime("%m%d%H%M")
    dumpname = f"convdom_{date}"
    fig_perturbed.savefig(dumpname + ".png")
    with open(dumpname + ".pkl", "wb") as f:
        pickle.dump(fig_perturbed, f)
