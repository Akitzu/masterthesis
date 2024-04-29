from pyoculus.problems import AnalyticCylindricalBfield
from pyoculus.integrators import RKIntegrator
from horus.convergence_domain import convergence_domain, plot_convergence_domain
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import datetime
import pickle


def trace_M(fun, rtol = 1e-10, initpoint = None):
    iparams = dict()
    iparams["rtol"] = rtol
    iparams["ode"] = fun

    integrator = RKIntegrator(iparams)

    if initpoint is None:
        raise ValueError("initpoint is not set")

    ic = np.array([initpoint[0], initpoint[1], 1.0, 0.0, 0.0, 1.0], dtype=np.float64)
    integrator.set_initial_value(0, ic)
    try:
        M = integrator.integrate(2*np.pi)
    except:
        return np.nan
    M = M[2:6].reshape((2,2)).T

    eigenvalues = np.sort(np.linalg.eigvals(M))
    return eigenvalues[0] / eigenvalues[1]

if __name__ == "__main__":
    ### Creating the pyoculus problem object
    print("\nCreating the pyoculus problem object\n")
    
    separatrix = {"type": "circular-current-loop", "amplitude": -10, "R": 6, "Z": -5.5}

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
        tol=1e-9,
    )
    # pyoproblem = AnalyticCylindricalBfield(6, 0, 0.91, 0.6, perturbations_args=[separatrix])

    ### Convergence domain calculation
    n1, n2 = 100, 100
    # rw = np.linspace(3.05, 3.15, n1)
    # zw = np.linspace(-1.7, -1.6, n2)
    # rw = np.linspace(3.0, 3.2, n1)
    # zw = np.linspace(-1.75, -1.5, n2)
    rw = np.linspace(3.5, 9, n1)
    zw = np.linspace(-6, 2, n2)

    RZs = np.meshgrid(rw, zw)

    ### Linearized error
    traces = [np.abs(trace_M(pyoproblem.f_RZ_tangent, initpoint = [r, z])) for r, z in zip(RZs[0].flatten(), RZs[1].flatten())]
    
    ### Plotting
    fig_perturbed = pickle.load(open("../poincare_04170914.pkl", "rb"))
    ax_perturbed = fig_perturbed.get_axes()[0]
    
    # mesh = ax_perturbed.pcolormesh(RZs[0], RZs[1], np.array(traces).reshape((n2, n1)), shading='nearest', cmap='viridis', alpha = 0.5)
    # fig_perturbed.colorbar(mesh, ax=ax_perturbed)

    # # Calculate the 5th and 95th percentiles of the data
    vmin, vmax = np.percentile(np.array(traces).reshape((n2, n1)), [5, 95])
    # Create a custom normalization object
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    mesh = ax_perturbed.pcolormesh(RZs[0], RZs[1], np.array(traces).reshape((n2, n1)), shading='nearest', cmap='viridis', alpha = 0.5, norm=norm)

    # Add colorbar
    cbar = fig_perturbed.colorbar(mesh, ax=ax_perturbed)

    fig_perturbed.set_size_inches(10, 6)
    date = datetime.datetime.now().strftime("%m%d%H%M")
    dumpname = f"ratio_M_{date}"
    with open(dumpname + ".pkl", "wb") as f:
        pickle.dump(fig_perturbed, f)

    plt.show()
    fig_perturbed.savefig(dumpname + ".png")