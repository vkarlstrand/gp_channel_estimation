""" gpr2d1_predict_freq.py

    Gaussian Process Regression (GPR)
    * Two GPs for each output.
    * Two dimensional input:    Frequency (f) and time (t).
    * One dimensional output:   Real part (a) and imaginary part (b).
    * Assumptions:
        -   Real and imaginary part are considered independent.

    Description:
    Uses GP to estimate transfer functions of 4 time-slices with different
    proportions of training data and test data. This to try and evaluate how many
    training data points are needed to approximate the transfer function.

    Setup: Python 3.6.9, GpPyTorch v1.0.0

"""



""" IMPORTS """
import numpy as np
import matplotlib.pyplot as plt
import torch
import gpytorch

from Data import Data # for importing wifi data, own class
from Data import get_mesh_x, get_mesh_y
from mpl_toolkits.mplot3d import Axes3D
import matplotlib



import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch
import gpytorch


rmse_y = [0.0244,
          0.0344,
          0.0456,
          0.0813,
          0.1601,
          0.1782,
          0.1460]

rmse_x =  [26/52, 17/52, 13/52, 11/52, 9/52, 8/52, 7/52]
rmse_x = [7, 6, 5, 4, 3, 2, 1]
rmse_x_ticks =  ["$\\frac{26}{52}$", "$\\frac{17}{52}$", "$\\frac{13}{52}$", "$\\frac{11}{52}$",
        "$\\frac{9}{52}$", "$\\frac{8}{52}$", "$\\frac{7}{52}$"]


matplotlib.rcParams.update({'font.size': 14})

fig = plt.figure(figsize=(8, 3.5))
# plt.suptitle("RMSE of Test Data")
ax = fig.add_subplot()
ax.plot(rmse_x, rmse_y,
                 c='k', linestyle='-', lw=1.5)
# ax.legend()
# ax.set_xlim([1, 6])
# ax.set_ylim([0, 1.25])
ax.set_xlabel("Proportion $k_{TR} $")
ax.set_ylabel("RMSE")
ax.set_yscale('log')
ax.grid(which="minor", alpha=0.25)
ax.grid(which="major", alpha=0.5)
ax.set_yticks([0, 0.025, 0.05, 0.1, 0.2, 0.3])
ax.set_yticklabels(["0", "0.025", "0.05", "0.1", "0.2", "0.3"])
ax.set_ylim([0.01, 0.3])
ax.set_xticks(rmse_x)
ax.set_xticklabels(rmse_x_ticks, fontsize=20)

fig.tight_layout(pad=3.0)
plt.show()