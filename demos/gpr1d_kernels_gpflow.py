# INSTRUCTION
# The following is done in virtual environment "tf_gpflow" (see README.md).
# The purpose of this file is to show how kernels can be combined to
# predict data better, considering different appearances of the data.
# This code is inspired by guides from GPflow.


# IMPORTS
import numpy as np
import tensorflow as tf
import gpflow
from gpflow.utilities import print_summary

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D


# DEBUG
DEBUG = True
PRINT = False


# FUNCTIONS
def h(x):
    return x*(1 + x/1.5) + 300*np.cos(2/3*np.pi*x)

def objective_closure():
    return - m.log_marginal_likelihood()


# PARAMETERS
data_bound = 25                                     # x upper bound
data_num = 80                                       # x steps
pred_bound = 50                                    # prediction upper bound
pred_num = 500                                      # prediction steps
noise = 32                                           # amount of noise
xmax = 50                                          # xlim for upper, when graphing
xmin = 0
ymax = 2400                                         # ylim for upper, when graphing
ymin = -400


# GENERATE DATA
x = np.linspace(0, data_bound, data_num)            # create array with range(0,data_bound) and data_num steps
y = h(x)                                            # generate function values for x
for i in range(len(y)):                             # add noise y-values
        y[i] += np.random.normal(0, noise)
X0 = np.asarray(x, dtype=np.float64).reshape(-1,1)  # convert to numpy array with type float64 and reshapte to (data_num * 1)-vector...
Y0 = np.asarray(y, dtype=np.float64).reshape(-1,1)
X = tf.cast(X0, dtype=tf.float64)                   # convert to tensors, just in case..?
Y = tf.cast(Y0, dtype=tf.float64)
if PRINT:
    print("X: ", X)
    print("Y: ", Y)


# CREATE FIGURE
fig1 = plt.figure(1, figsize = (8, 3))              # figure to plot data
plt.rc('font', size=12)
plt.plot(X, Y, 'k.', alpha=0.5, mew=2, label = 'Data with noise'); # plot training data


# KERNELS TO COMBINE TO 'k'
k1 = gpflow.kernels.RBF()
k2 = gpflow.kernels.Linear()
k3 = gpflow.kernels.Periodic(k1, period=3.0)
k4 = gpflow.kernels.Polynomial(degree=2.0)
k = k1 + k2 + k3 + k4
k_title = "RBF + Linear + Periodic + Polynomial"
if DEBUG:
    print_summary(k)


# MODEL
m = gpflow.models.GPR(data=(X, Y), kernel = k, mean_function = None)
#m.likelihood.variance.assign(0.001)   # to set when only RBF is used
#m.kernel.lengthscale.assign(1)
#m.kernel.variance.assign(8)
opt = gpflow.optimizers.Scipy()
opt_logs = opt.minimize(objective_closure, m.trainable_variables, options=dict(maxiter=50))
if DEBUG:
    print_summary(m)
    print("Trainable variables: ", m.trainable_variables)
    print("Opt logs: ", opt_logs)


# PREDICT
xx = np.linspace(0, pred_bound, pred_num).reshape(-1, 1)    # create data space to predict
mean, var = m.predict_f(xx)
samples = m.predict_f_samples(xx, 10)               # sample some functions


# # GRAPH
# plt.title(k_title)
# plt.plot(xx, mean, 'k--', lw=2, label = 'Mean estimate') # plot mean estimate
# plt.fill_between(xx[:,0],
#                  mean[:,0] - 1.96 * np.sqrt(var[:,0]),  # plot variance estimate
#                  mean[:,0] + 1.96 * np.sqrt(var[:,0]),
#                  color='k', alpha=0.25, label = 'Variance estimate (95% confidence)')
# # plt.plot(xx, samples[:, :, 0].numpy().T, 'k', linewidth=.5)   # plot sample functions
# plt.xlim(xmin, xmax)
# plt.ylim(ymin, ymax)
# # plt.legend()

# Get covariance matrix of posterior.
cmap = 'Greys_r'
samples_num = 200
xx_posterior_cov = np.linspace(0, pred_bound, samples_num).reshape(samples_num, 1)
K_posterior_mean, K_posterior_cov = m.predict_f(xx_posterior_cov, full_cov=True)
K_max = np.amax(K_posterior_cov)
K_min = np.amin(K_posterior_cov)
print(K_posterior_cov)
K_posterior_cov = K_posterior_cov - K_min + 10
# K_posterior_cov /= np.amax(K_posterior_cov)
print(K_posterior_cov)
print(np.amin(K_posterior_cov))
print(np.amax(K_posterior_cov))
K_posterior_cov = np.log(K_posterior_cov)

# plt.figure(3, figsize =(8,4))
fig2, (ax2) = plt.subplots(1, figsize = (4, 4))              # figure to plot data
plt.rc('font', size=12)
ax2.pcolormesh(K_posterior_cov[0], cmap=cmap)   # for some reason 'm_posterior.predict_f' returns the grid in a "extra" [], which can't be plotted
ax2.set_xticks(np.linspace(0, samples_num, 6))
ax2.set_xticklabels([0, 20, 40, 60, 80, 100])
ax2.set_yticks(np.linspace(0, samples_num, 6))
ax2.set_yticklabels([0, 20, 40, 60, 80, 100])
plt.gca().invert_yaxis()
fig2.suptitle("Covariance matrix")


true = h(xx)

matplotlib.rcParams.update({'font.size': 18})
fig = plt.figure(figsize=(10,4))
gs = fig.add_gridspec(1, 3)
ax1 = fig.add_subplot(gs[0, 0:2])
ax1.scatter(X, Y, s=20, alpha=1, edgecolors='k', facecolors='none', label = 'Observations') # plot training data
ax1.plot(xx, mean, 'k', linestyle='-', lw=1.5, label = 'Mean') # plot mean estimate
ax1.plot(xx, true, color='r', linestyle='-', lw=0.5, label = 'True function') # plot mean estimate
ax1.fill_between(xx[:,0],
                 mean[:,0] - 1.96 * np.sqrt(var[:,0]),  # plot variance estimate
                 mean[:,0] + 1.96 * np.sqrt(var[:,0]),
                 color='k', alpha=0.2, label = 'Variance')
#plt.plot(xx, samples[:, :, 0].numpy().T, 'k', linewidth=.5)   # plot sample functions
ax1.set_xlim(xmin, xmax)
ax1.set_ylim(ymin, ymax)
ax1.set_xlabel("$x$")
ax1.set_ylabel("$y$")
ax1.legend(ncol=2)
ax1.set_title("GPR with combined kernels")

ax2 = fig.add_subplot(gs[0, 2:3])
cmap = 'Greys_r'
ax2.pcolormesh(K_posterior_cov[0], cmap=cmap)   # for some reason 'm_posterior.predict_f' returns the grid in a "extra" [], which can't be plotted
#ax2.set_xticks(np.linspace(0, samples_num, 6))
ax2.set_xticklabels([])
#ax2.set_yticks(np.linspace(0, samples_num, 6))
ax2.set_yticklabels([])
ax2.set_xlabel("$x_j$")
ax2.set_ylabel("$x_i$")
plt.gca().invert_yaxis()
ax2.set_title("Covariance matrix")

fig.tight_layout(pad=1)


# PLOT ALL FIGURES
plt.show()
