from horus import ncsx, convergence_domain, plot_convergence_domain
from pyoculus.problems import SimsoptBfieldProblem
import matplotlib.pyplot as plt
import numpy as np
import pickle

import datetime
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

bs, bsh, (nfp, coils, ma, sc_fieldline) = ncsx()
logging.info("CONFIGURATION LOADED")

pyoproblem = SimsoptBfieldProblem(ma.gamma()[0, 0], ma.gamma()[0, 2], nfp, bs)

## Recover the Poincare plot
fig = pickle.load(open("../../code/output/NCSX_poincare.pkl", "rb"))
ax = fig.get_axes()[0]

### Convergence domain calculation
rw = np.linspace(1.2, 1.8, 3)
zw = np.linspace(0, 0.7, 3)

logging.info(f"rw: [{rw[0]}, {rw[-1]}]")
logging.info(f"zw: [{zw[0]}, {zw[-1]}]")

convdom = convergence_domain(
    pyoproblem,
    rw,
    zw,
    pp=3,
    qq=7,
    sbegin=0.5,
    send=3,
    tol=1e-8,
    checkonly=True,
    eps=1e-4,
    rtol=1e-10,
)

logging.info(f"Computation done")

convdomplot = convdom[0:4]
plot_convergence_domain(*convdomplot, ax)

# Save the result
date = datetime.datetime.now().strftime("%m%d%H%M")
dumpname = f"convergence_domain_{date}.pkl"
with open(dumpname, "wb") as f:
    pickle.dump(fig, f)