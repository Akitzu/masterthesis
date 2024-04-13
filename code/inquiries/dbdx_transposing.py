from pyoculus.problems import CylindricalBfield, AnalyticCylindricalBfield
from pyoculus.solvers import PoincarePlot, FixedPoint, Manifold
import matplotlib.pyplot as plt
import numpy as np

separatrix = {"type": "circular-current-loop", "amplitude": -4.2, "R": 3, "Z": -2.2}
ps_cyl = AnalyticCylindricalBfield(3, 0, 0.91, 0.7, perturbations_args = [separatrix])

ps_cyl.dBdX([3., 0., 0.1])
ps_cyl.f_RZ_tangent(0., [3., 0.1, 1, 0, 0, 1])