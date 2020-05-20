""" kernel_covariance_matrix.py

    Computes the covariance matrix with a RBF kernel and prints covariance matrix.

"""



""" IMPORTS
"""
import numpy
import matplotlib.pyplot as plt
import matplotlib

import scipy
import sklearn.metrics.pairwise
import torch
import gpytorch



""" GENERATE DATA
    Create data.
"""
def func(x, shift, noise):
    y = numpy.sinc(x/2 + shift) + 0.04*x
    for i in range(len(y)):
        y[i] += noise*numpy.random.randn(1)
    return y


# Generate true function
x = numpy.linspace(-15, 15, 400)
y = func(x, shift=2.2, noise=0)

# Some covariance matrices.
yy_cov = sklearn.metrics.pairwise.rbf_kernel(y.reshape(-1,1))

matplotlib.rcParams.update({'font.size': 24})

fig_cov = plt.figure(figsize=(10, 5))
gs_cov = fig_cov.add_gridspec(1,2)

cmap = 'Greys_r'

ax_cov_yy = fig_cov.add_subplot(gs_cov[0,1])
ax_cov_yy.set_title("Kernel matrix \n $k_{RBF}(\mathbf{x}_i, \mathbf{x}_j)$")
ax_cov_yy.pcolormesh(yy_cov, cmap=cmap)
plt.gca().invert_yaxis()


ax_func = fig_cov.add_subplot(gs_cov[0,0])
ax_func.set_title("Data points \n $\mathbf{x}_n = (x_{n,1}, x_{n,2})$")
ax_func.plot(x, y, c='k', lw=3)
ax_func.set_xlim([-15, 15])
ax_func.set_ylim([-1, 1])
ax_func.set_xlabel("$x_1$")
ax_func.set_ylabel("$x_2$")

plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[]);

plt.tight_layout(pad=1.1)
plt.show()