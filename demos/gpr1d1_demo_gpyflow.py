# INSTRUCTIONS
# This demo uses GPflow and is done in the virtual environment "tf_gpflow", see README.md of main repository.
# The demo is heavily based on the guide provided by GPflow at:
#   https://nbviewer.jupyter.org/github/GPflow/GPflow/blob/develop/doc/source/notebooks/basics/regression.ipynb


# IMPORTS
import gpflow
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from gpflow.utilities import print_summary

from scipy.stats import kde



# GENERATE DATA
# Five observed data points posterior being conditioned on.
x = [[1], [3], [4.5]]
y = [[1], [-3.75], [0.5]]
X = np.asarray(x, dtype = np.float64)       # convert list to numpy arrays...
Y = np.asarray(y, dtype = np.float64)

# For prior, with one "faked" observations out of the plot range.
# Needed so that prediction function can be used to sample from this "faked" prior.
x0 = [[-100]]
y0 = [[0]]
X0 = np.asarray(x0, dtype = np.float64)     # convert list to numpy arrays...
Y0 = np.asarray(y0, dtype = np.float64)


# MODEL AND HYPERPARAMETERS
k = gpflow.kernels.RBF()
k_prior_lengthscale = 0.65  # These values are very close to to the optimal ones.
k_prior_variance = 1.5

# PRIOR
# Set model and its parameters.
m_prior = gpflow.models.GPR(data=(X0, Y0), kernel = k, mean_function = None)
m_prior.kernel.lengthscale.assign(k_prior_lengthscale)
m_prior.kernel.variance.assign(k_prior_variance)
# Generate test points for prediction.
xx_prior = np.linspace(0, 5, 100).reshape(100, 1)
# Predict mean and variance for test points.
mean_prior, var_prior = m_prior.predict_f(xx_prior)
# Generate 5 samples from prior.
samples_prior = m_prior.predict_f_samples(xx_prior, 5)


# # Plot prior.
# matplotlib.rcParams.update({'font.size': 14})
# plt.figure(1, figsize = (8, 3))
# pm = plt.plot(xx_prior, mean_prior, '--k', lw=2)
# pv = plt.fill_between(xx_prior[:,0],                                # create bounderies of variance and fill inbetween
#                  mean_prior[:,0] - 1.96 * np.sqrt(var_prior[:,0]),  # k = 1.96 (95% confidence)
#                  mean_prior[:,0] + 1.96 * np.sqrt(var_prior[:,0]),
#                  color='gray', alpha=0.2)
# plt.title("Prior")
# plt.plot(xx_prior, samples_prior[:, :, 0].numpy().T, 'k', linewidth=0.5)
# plt.xlim(0, 5)
# plt.ylim(-3,3)


# POSTERIOR
# Set model and its parameters.
m_posterior = gpflow.models.GPR(data=(X, Y), kernel = k, mean_function = None)
m_posterior.likelihood.variance.assign(0.001)               # gives tight variance around observed data points
m_posterior.kernel.lengthscale.assign(k_prior_lengthscale)
m_posterior.kernel.variance.assign(k_prior_variance)

# Optmize model parameters (comment in to optimize the parameters above, for now they are set appropriately from the start)
# opt = gpflow.optimizers.Scipy()
# def objective_closure():
#     return - m_posterior.log_marginal_likelihood()
# opt_logs = opt.minimize(objective_closure, m_posterior.trainable_variables, options=dict(maxiter=100))
# print_summary(m_posterior)

# Generate test points for prediction.
xx_posterior = np.linspace(0, 5, 100).reshape(100, 1)
# Predict mean and variance for test points.
mean_posterior, var_posterior = m_posterior.predict_f(xx_posterior)
# Generate 5 samples from posterior.
samples_posterior = m_posterior.predict_f_samples(xx_posterior, 5)

# # Plot posterior.
# plt.figure(2, figsize = (8, 3))
# matplotlib.rcParams.update({'font.size': 14})
# pp = plt.plot(X, Y, 'ko', mew=2, label = 'Data')
# pm = plt.plot(xx_posterior, mean_posterior, '--k', lw=2)
# pv = plt.fill_between(xx_posterior[:,0],                            # create bounderies of variance and fill inbetween
#                  mean_posterior[:,0] - 1.96 * np.sqrt(var_posterior[:,0]),
#                  mean_posterior[:,0] + 1.96 * np.sqrt(var_posterior[:,0]),
#                  color='gray', alpha=0.2)
# plt.title("Posterior")
# plt.plot(xx_posterior, samples_posterior[:, :, 0].numpy().T, 'k', linewidth=0.5)
# plt.xlim(0, 5)
# plt.ylim(-3,3)



# # Get covariance matrix of posterior and prior.
samples_num = 50
xx_posterior_cov = np.linspace(0, 5, samples_num).reshape(samples_num, 1)
xx_prior_cov = np.linspace(0, 5, samples_num).reshape(samples_num, 1)
K_posterior_mean, K_posterior_cov = m_posterior.predict_f(xx_posterior_cov, full_cov=True)
K_prior_mean, K_prior_cov = m_prior.predict_f(xx_prior_cov, full_cov=True)
# # plt.figure(3, figsize =(8,4))
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9,4))
# matplotlib.rcParams.update({'font.size': 14})
# plt.rc('font', size=16)
# cmap = 'Greys_r'
# ax1.pcolormesh(K_prior_cov[0], cmap=cmap)       # for some reason 'm_posterior.predict_f' returns the grid in a "extra" [], which can't be plotted
# ax1.invert_yaxis()
# ax2.pcolormesh(K_posterior_cov[0], cmap=cmap)
# ax2.invert_yaxis()
# ax1.set_xticks(np.linspace(0, samples_num, 6))
# ax1.set_xticklabels([0, 1, 2, 3, 4, 5])
# ax1.set_yticks(np.linspace(0, samples_num, 6))
# ax1.set_yticklabels([0, 1, 2, 3, 4, 5])
# ax2.set_xticks(np.linspace(0, samples_num, 6))
# ax2.set_xticklabels([0, 1, 2, 3, 4, 5])
# ax2.set_yticks(np.linspace(0, samples_num, 6))
# ax2.set_yticklabels([0, 1, 2, 3, 4, 5])
# fig.suptitle("Covariance matrices")
# ax1.set_title("Prior")
# ax2.set_title("Posterior")
# fig.tight_layout(pad=2)


matplotlib.rcParams.update({'font.size': 19})
fig = plt.figure(figsize=(10,8))
gs = fig.add_gridspec(2, 3)
ax1 = fig.add_subplot(gs[1, 0:2])
ax1.scatter(X, Y, s=60, facecolors='none', edgecolors='k', label = 'Training data')
ax1.plot(xx_posterior, mean_posterior, 'k', lw=1.5, label="Mean")
ax1.fill_between(xx_posterior[:,0],                            # create bounderies of variance and fill inbetween
                 mean_posterior[:,0] - 1.96 * np.sqrt(var_posterior[:,0]),
                 mean_posterior[:,0] + 1.96 * np.sqrt(var_posterior[:,0]),
                 color='gray', alpha=0.2, label="Variance")
ax1.set_title("Posterior")
ax1.plot(xx_posterior, samples_posterior[1:-1, :, 0].numpy().T, 'k', linestyle="dotted", linewidth=1.5)
ax1.plot(xx_posterior, samples_posterior[0:1, :, 0].numpy().T, 'k', linestyle="dotted", linewidth=1.5, label="Sampled functions")
ax1.set_xlim(0, 5)
ax1.set_ylim(-5,5)
ax1.set_xlabel("$x$")
ax1.set_ylabel("$y$")
ax1.legend(ncol=2, loc='upper right')

ax2 = fig.add_subplot(gs[0, 0:2])
ax2.plot(xx_prior, mean_prior, 'k', lw=2, label="Mean")
ax2.fill_between(xx_prior[:,0],                            # create bounderies of variance and fill inbetween
                 mean_prior[:,0] - 1.96 * np.sqrt(var_prior[:,0]),
                 mean_prior[:,0] + 1.96 * np.sqrt(var_prior[:,0]),
                 color='gray', alpha=0.2, label="Variance")
ax2.set_title("Prior")
ax2.plot(xx_prior, samples_prior[1:-1, :, 0].numpy().T, 'k', linestyle="dotted", linewidth=1.5)
ax2.plot(xx_prior, samples_prior[0:1, :, 0].numpy().T, 'k', linestyle="dotted", linewidth=1.5, label="Sampled functions")
ax2.set_xlim(0, 5)
ax2.set_ylim(-5,5)
ax2.set_xlabel("$x$")
ax2.set_ylabel("$y$")
ax2.legend(ncol=2, loc='upper right')

cmap = 'Greys_r'
ax3 = fig.add_subplot(gs[0, 2:3])
ax4 = fig.add_subplot(gs[1, 2:3])
ax3.pcolormesh(K_prior_cov[0], cmap=cmap)       # for some reason 'm_posterior.predict_f' returns the grid in a "extra" [], which can't be plotted
ax3.invert_yaxis()
ax4.pcolormesh(K_posterior_cov[0], cmap=cmap)
ax4.invert_yaxis()
ax3.set_xticks(np.linspace(0, samples_num, 6))
ax3.set_xticklabels([0, 1, 2, 3, 4, 5])
ax3.set_yticks(np.linspace(0, samples_num, 6))
ax3.set_yticklabels([0, 1, 2, 3, 4, 5])
ax4.set_xticks(np.linspace(0, samples_num, 6))
ax4.set_xticklabels([0, 1, 2, 3, 4, 5])
ax4.set_yticks(np.linspace(0, samples_num, 6))
ax4.set_yticklabels([0, 1, 2, 3, 4, 5])
ax3.set_title("Prior covariance")
ax4.set_title("Posterior covariance")
ax3.set_xlabel("$x_j$")
ax3.set_ylabel("$x_i$")
ax4.set_xlabel("$x_j$")
ax4.set_ylabel("$x_i$")

fig.tight_layout(pad=0)


# POSTERIOR WITH DIFFERENT PARAMETER VALUES
# The different parameter values used in the paper.
k_prior_lengthscale1 = 0.2
k_prior_variance1 = 0.25
k_prior_lengthscale2 = 0.65
k_prior_variance2 = 1.5
k_prior_lengthscale3 = 1.5
k_prior_variance3 = 4

K_ls = [k_prior_lengthscale1, k_prior_lengthscale2, k_prior_lengthscale3]
K_v = [k_prior_variance1, k_prior_variance2, k_prior_variance3]

LS = ["Short", "Good", "Long"]
V = ["Small", "Good", "Large"]


matplotlib.rcParams.update({'font.size': 18})
fig = plt.figure(figsize=(10,8))
#fig.suptitle("Length-scale", fontsize="20")
gs = fig.add_gridspec(3, 3)
i = 0
for v in K_v:
    j = 0
    for ls in K_ls:
        ax = fig.add_subplot(gs[i, j])
        # Set model.
        m_posterior2 = gpflow.models.GPR(data=(X, Y), kernel = k, mean_function = None)
        m_posterior2.likelihood.variance.assign(0.001)
        m_posterior2.kernel.lengthscale.assign(ls)    # <-- choose length-scale
        m_posterior2.kernel.variance.assign(v)          # <-- choose variance

        # Optmize model parameters (comment in to optimize the parameters above, for now they are set appropriately from the start)
        # opt = gpflow.optimizers.Scipy()
        # def objective_closure():
        #     return - m_posterior.log_marginal_likelihood()
        # opt_logs = opt.minimize(objective_closure, m_posterior.trainable_variables, options=dict(maxiter=100))
        # print_summary(m_posterior)

        # Generate test points for prediction.
        xx_posterior2 = np.linspace(0, 5, 200).reshape(200, 1)  # higher resolution here, since the curves tends to be more "wiggly"
        # Predict mean and variance for test points.
        mean_posterior2, var_posterior2 = m_posterior2.predict_f(xx_posterior2)
        # Generate 5 samples from posterior.
        samples_posterior2 = m_posterior2.predict_f_samples(xx_posterior2, 5)

        # Plot posterior2.
        ax.scatter(X, Y, facecolors='none', edgecolors='k')
        ax.plot(xx_posterior2, mean_posterior2, 'k', lw=1.5)
        ax.fill_between(xx_posterior2[:,0],                           # create bounderies of variance and fill inbetween
                         mean_posterior2[:,0] - 1.96 * np.sqrt(var_posterior2[:,0]),
                         mean_posterior2[:,0] + 1.96 * np.sqrt(var_posterior2[:,0]),
                         color='gray', alpha=0.2)
        ax.plot(xx_posterior2, samples_posterior2[:, :, 0].numpy().T, 'k', linewidth=1, linestyle='dotted')
        ax.set_xticks([], [])
        ax.set_yticks([], [])
        ax.set_xlim(0, 5)
        ax.set_ylim(-5,5)
        if i == 0:
            ax.set_title(LS[j])
        if j == 0:
            ax.set_ylabel(V[i], fontsize="18")
        j = j + 1
    i = i + 1


# PLOT ALL FIGURES
plt.show()
