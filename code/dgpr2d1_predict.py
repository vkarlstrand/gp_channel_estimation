""" dgpr2d1_predict_freq.py

    Deep Gaussian Process Regression (DGPR)
    * Two DGPs for each output.
    * Two dimensional input:    Frequency (f) and time (t).
    * One dimensional output:   Real part (a) and imaginary part (b).
    * Assumptions:
        -   Real and imaginary part are considered independent.

    Description:
    Uses DGP to estimate transfer functions of 4 time-slices with different
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



""" IMPORT DATA """
# PARAMETERS FOR IMPORT
LIMIT = 5 # number of time samples to import
FILE = "data/wifi_capture_190331.csv"
JUMP = 0 # number of time samples to jumo over during each import step
SKIP = 160 # number of time samples to skip in the beginning
SKIP_CLOSE = 0.25 # difference in seconds between samples to skip if close to each other
MAC_ID = "ff:ff:ff:ff:ff:ff" # MAC id to import from file

FILEs = ["data/wifi_capture_190331.csv" , "data/wifi_capture_190413.csv",
         "data/wifi_capture_20190329.csv", "data/wifi_capture_20190331_in2outdoor.csv",
         "data/wifi_capture_20190412.csv"]
MAC_IDs = ["8c:85:90:a0:2a:6f", "1c:b7:2c:d7:67:ec", "ff:ff:ff:ff:ff:ff",
           "34:a3:95:c3:57:10", "a0:93:51:72:24:0f"]
LIMIT = 5 # number of time samples to import
JUMP = 0 # number of time samples to jump over after each imported time sample
SKIP = 160 # number of time samples to skip over in the beginning
SKIP_CLOSE = 0.25 # if time sample is within SKIP_CLOSE to previous time sample imported, skip it
MAC_ALL = False # import all kinds
FILE = FILEs[0] # file to import from
MAC_ID = MAC_IDs[2]
DATA_SET = "DGPR - Every other sample as training data"

# D1: 03 - JUMP=0 SKIP=0 SKIP_CLOSE=0
# D2: 12 - JUMP=0 SKIP=500 SKIP_CLOSE=0
# D3: 13 - JUMP0 SKIP=0 SKIP_CLOSE=0
# D4: 20 - JUMP0 SKIP=0 SKIP_CLOSE=0
# D5: 41 - JUMP0 SKIP=1000 SKIP_CLOSE=0

data = Data("DGPR2D1") # initalize data object
data.load(csv_file=FILE, # load data
          mac_str=MAC_ID,
          limit=LIMIT,
          skip=SKIP,
          jump=JUMP,
          skip_close=SKIP_CLOSE)
data.prep() # prepare data and put into vectors
data.norm_effect() # normalize with respect to of absolute value over all transfer functions in time
data.norm() # normalize all data to be within (-1,1)

# GET DATA
X_np = data.get_x()
Y_np = data.get_y() # or data.get_y_abs_ang()
y1_np = Y_np[:,0] # get first column of real (a) values
y2_np = Y_np[:,1] # get second column of imaginary (b) values

# # CONVERT TO TENSORS FOR TRAINING AND TEST DATA
X = torch.from_numpy(X_np).type('torch.FloatTensor').contiguous()
y1 = torch.from_numpy(y1_np).type('torch.FloatTensor').contiguous()
y2 = torch.from_numpy(y2_np).type('torch.FloatTensor').contiguous()

# Split into train and test data in proportions TRAIN and TEST.
TRAIN = 1
TEST = 4
X_train, X_test = data.split(X_np, TRAIN, TEST)
Y_train, Y_test = data.split(Y_np, TRAIN, TEST)

# # For all as training and test data.
# X_train = X_np
# X_test = X_np
# Y_train = Y_np
# Y_test = Y_np

# Convert to tensors.
X_train = torch.from_numpy(X_train).type('torch.FloatTensor').contiguous()
X_test = torch.from_numpy(X_test).type('torch.FloatTensor').contiguous()
y1_train = torch.from_numpy(Y_train[:,0]).type('torch.FloatTensor').contiguous()
y1_test = torch.from_numpy(Y_test[:,0]).type('torch.FloatTensor').contiguous()
y2_train = torch.from_numpy(Y_train[:,1]).type('torch.FloatTensor').contiguous()
y2_test = torch.from_numpy(Y_test[:,1]).type('torch.FloatTensor').contiguous()



""" PLOT DATA BEFORE TRAINING
    Plot 3D graph of X (frequency and time) and y (real, imaginary, absolute or angle)
"""
# PREPARE MESHES FOR PLOTTING
mesh_f, mesh_t = get_mesh_x(tensor=X, f_num=data.get_f_num(), t_num=data.get_t_num())
mesh_y1 = get_mesh_y(tensor=y1, f_num=data.get_f_num(), t_num=data.get_t_num())
mesh_y2 = get_mesh_y(tensor=y2, f_num=data.get_f_num(), t_num=data.get_t_num())

mesh_f_train = X_train[:,0]
mesh_t_train = X_train[:,1]

mesh_f_test = X_test[:,0]
mesh_t_test = X_test[:,1]

mesh_y1_train = y1_train
mesh_y2_train = y2_train

mesh_y1_test = y1_test
mesh_y2_test = y2_test


# CREATE FIGURE
marksize = 80
matplotlib.rcParams.update({'font.size': 24})
fig_data = plt.figure(figsize=(20, 8))
plt.suptitle(DATA_SET)

# PLOT DATA FOR 1
ax_data1 = fig_data.add_subplot(1, 2, 1, projection="3d")
ax_data1.set_title("Real part")
ax_data1.plot_surface(mesh_f, mesh_t, mesh_y1,
                 cmap="jet", alpha=0.35)
ax_data1.plot_wireframe(mesh_f, mesh_t, mesh_y1,
            rstride=1, cstride=0, color='k', linestyle='-', alpha=0.35)
ax_data1.scatter(mesh_f_train, mesh_t_train, mesh_y1_train,
            marker='o', label="Training data", s=marksize, facecolor=(0,0,0,0), edgecolor=(0,0,0,0.65), linewidth=2)
ax_data1.scatter(mesh_f_test, mesh_t_test, mesh_y1_test,
            color='k', marker='*', label="Test data", alpha=0.35, s=marksize)

# PLOT INFO
ax_data1.set_xlabel("$f \ [kHz]$")
ax_data1.set_ylabel("$t \ [s]$")
ax_data1.set_zlabel("$\mathcal{Re} \ (H(f,t))$")
freqs = data.get_frequencies()
ax_data1.set_xticks([-1, 0, 1])
ax_data1.set_xticklabels([str(freqs[0]), "0", str(freqs[-1])])
time_str = str(np.around((data.get_time()[-1]+1)/2 * data.get_last_time(), 3))
ax_data1.set_yticks([data.get_time()[-0], data.get_time()[-1]])
ax_data1.set_yticklabels(["0", time_str])


# PLOT DATA FOR 2
ax_data2 = fig_data.add_subplot(1, 2, 2, projection="3d")
ax_data2.set_title("Imaginary part")
ax_data2.plot_surface(mesh_f, mesh_t, mesh_y2,
                 cmap="jet", alpha=0.35)
ax_data2.plot_wireframe(mesh_f, mesh_t, mesh_y2,
            rstride=1, cstride=0, color='k', linestyle='-', alpha=0.35)
ax_data2.scatter(mesh_f_train, mesh_t_train, mesh_y2_train,
            marker='o', label="Training data", s=marksize, facecolor=(0,0,0,0), edgecolor=(0,0,0,0.65), linewidth=2)
ax_data2.scatter(mesh_f_test, mesh_t_test, mesh_y2_test,
            color='k', marker='*', label="Test data", alpha=0.35, s=marksize)

# PLOT INFO
ax_data2.set_xlabel("$f \ [kHz]$")
ax_data2.set_ylabel("$t \ [s]$")
ax_data2.set_zlabel("$\mathcal{Im} \ (H(f,t))$")
freqs = data.get_frequencies()
ax_data2.set_xticks([-1, 0, 1])
ax_data2.set_xticklabels([str(freqs[0]), "0", str(freqs[-1])])
time_str = str(np.around((data.get_time()[-1]+1)/2 * data.get_last_time(), 3))
ax_data2.set_yticks([data.get_time()[-0], data.get_time()[-1]])
ax_data2.set_yticklabels(["0", time_str])

# ax_data1.set_zticks([-1, 0, 1])
# ax_data1.set_zticklabels(["-1","0", "1"])
#
# ax_data2.set_zticks([-1, 0, 1])
# ax_data2.set_zticklabels(["-1","0", "1"])

fig_data.tight_layout(pad=3)
ax_data1.xaxis.labelpad=30
ax_data1.yaxis.labelpad=30
ax_data1.zaxis.labelpad=30
ax_data2.xaxis.labelpad=30
ax_data2.yaxis.labelpad=30
ax_data2.zaxis.labelpad=30

ax_data1.legend(loc='lower center', bbox_to_anchor=(0.5, -0.18),
    fancybox=True, shadow=False, ncol=3)
ax_data2.legend(loc='lower center', bbox_to_anchor=(0.5, -0.18),
    fancybox=True, shadow=False, ncol=3)




""" DGP CLASS """
# IMPORT FOR EASIER CODING BELOW
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution
from gpytorch.distributions import MultivariateNormal
from gpytorch.models import ApproximateGP, GP
from gpytorch.mlls import VariationalELBO, AddedLossTerm
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.models.deep_gps import DeepGPLayer, DeepGP
from gpytorch.mlls import DeepApproximateMLL

class DGPHiddenLayer(DeepGPLayer):
    def __init__(self, input_dims, output_dims, num_inducing=128, mean_type='constant'):
        if output_dims is None:
            inducing_points = torch.randn(num_inducing, input_dims)
            batch_shape = torch.Size([])
        else:
            inducing_points = torch.randn(output_dims, num_inducing, input_dims)
            batch_shape = torch.Size([output_dims])

        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=num_inducing,
            batch_shape=batch_shape)

        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True)

        super(DGPHiddenLayer, self).__init__(variational_strategy, input_dims, output_dims)

        if mean_type == 'constant':
            self.mean_module = ConstantMean(batch_shape=batch_shape)
        else: # (if 'linear')
            self.mean_module = LinearMean(input_dims)

        #lengthscale_constraint = gpytorch.constraints.Interval(0.0001, 10.0) # needs to be floats
        lengthscale_prior = gpytorch.priors.NormalPrior(0.1, 2.0)
        outputscale_prior = gpytorch.priors.NormalPrior(1.0, 3.0)
        lengthscale_constraint = None
        #lengthscale_prior = None
        self.covar_module = ScaleKernel(
            RBFKernel(batch_shape=batch_shape,
                      ard_num_dims=input_dims,
                      #active_dims=(0),
                      lengthscale_constraint=lengthscale_constraint,
                      lengthscale_prior=lengthscale_prior),

                  outputscale_prior=outputscale_prior,
                  batch_shape=batch_shape,
                  ard_num_dims=input_dims)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class DGP(DeepGP):
    def __init__(self, train_x_shape):
        hidden_layer = DGPHiddenLayer(
            input_dims=train_x_shape[-1],
            output_dims=NUM_OUTPUT_DIMS,
            num_inducing=NUM_INDUCING,
            mean_type='linear')

        last_layer = DGPHiddenLayer(
            input_dims=hidden_layer.output_dims,
            output_dims=None,
            num_inducing=NUM_INDUCING,
            mean_type='linear')

        super().__init__()

        self.hidden_layer = hidden_layer
        self.last_layer = last_layer
        self.likelihood = GaussianLikelihood()

    def forward(self, inputs):
        hidden_layer_output = self.hidden_layer(inputs)
        output = self.last_layer(hidden_layer_output)
        return output

    def optimize(self, X, y, epochs=500, lr=0.01, samples=10):
        opt = torch.optim.Adam([{'params': self.parameters()}], lr=lr)
        mll = DeepApproximateMLL(VariationalELBO(self.likelihood, self, X.shape[-2]))
        self.train()
        self.likelihood.train()
        lls = []
        gpytorch.settings.skip_posterior_variances(state=False)
        for i in range(epochs):
            with gpytorch.settings.num_likelihood_samples(samples):
                opt.zero_grad()
                output = self(X)
                loss = -mll(output, y)
                loss.backward()
                noise = self.likelihood.noise.item()
                print('Iter: %d/%d,   Loss: %.5f,   Likelihood noise: %.3f' % (
                     i+1, epochs, loss.item(), noise))
                lls.append(loss.item())
                opt.step()
        return lls

    def print_ls(self):
        print("Hidden layer lengtscale:", model.hidden_layer.covar_module.base_kernel.lengthscale.detach().numpy())
        print("Last layer lengtscale:", model.last_layer.covar_module.base_kernel.lengthscale.detach().numpy())

    def predict(self, Xp, samples=10):
        self.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.num_likelihood_samples(samples):
            # preds_marg = self.likelihood(model(Xp))
            # means_marg = preds_marg.mean
            # vars_marg = preds_marg.variance
            # lowers_marg, uppers_marg = preds_marg.confidence_region()
            # mean_marg = means_marg.mean(0)
            # var_marg = vars_marg.mean(0)
            # lower_marg = lowers_marg.mean(0)
            # upper_marg = uppers_marg.mean(0)

            preds = self(Xp)
            means = preds.mean
            mean = means.mean(0)
            vars = preds.variance
            var = vars.mean(0)
            vars_lower, vars_upper = preds.confidence_region()
            var_lower = vars_lower.mean(0)
            var_upper = vars_upper.mean(0)

            cov_mat = preds.covariance_matrix
            cov_mat = cov_mat.mean(0)

            return mean, var, var_lower, var_upper, cov_mat



""" MODEL AND TRAINING
    Sometimes it seems that the loss is just constant, but with many iterations,
    hopefully it goes done eventuelly quite quickly.
    Sometimes it's just bad, then try and rerun and see if it goes better.
"""
# Reshape training data to [num_training_points, dimensions].
X_train = X_train.reshape(-1,2)

# Set parameters for training.
EPOCHS = 1500
LEARNING_RATE = 0.0075
SAMPLES = 64
NUM_INDUCING = 256
NUM_OUTPUT_DIMS = 1

# Initialize model with the x.shape, requires data on form [num_training_points, dimension]
model1 = DGP(X_train.shape)
model2 = DGP(X_train.shape)
lls1 = model1.optimize(X=X_train, y=y1_train, epochs=EPOCHS, lr=LEARNING_RATE, samples=SAMPLES)
lls2 = model2.optimize(X=X_train, y=y2_train, epochs=EPOCHS, lr=LEARNING_RATE, samples=SAMPLES)



""" PREDICTIONS """
# Generate prediction input data to be able to plot mean and variance as a continous function.
PRED_F_STEPS = 50
PRED_T_STEPS = 50

Xp_f = torch.linspace(-1.25, 1.25, PRED_F_STEPS).reshape(-1,1) # use some range close to training, but prehaps a bit outside to see deviation where uncertain
Xp_t = torch.linspace(-1.25, 1.25, PRED_T_STEPS).reshape(-1,1)
mesh_fp, mesh_tp = np.meshgrid(Xp_f, Xp_t)
Xp = np.dstack([mesh_fp, mesh_tp]).reshape(-1,2)
Xp = torch.from_numpy(Xp).type('torch.FloatTensor').contiguous()

# Make predictions of conditiona/posterior distrubution
mean1, var1, var_lower1, var_upper1, cov_mat1 = model1.predict(Xp=Xp, samples=SAMPLES)
mean2, var2, var_lower2, var_upper2, cov_mat2 = model2.predict(Xp=Xp, samples=SAMPLES)


""" TEST """
mean_test1, var_test1, var_lower_test1, var_upper_test1, cov_mat_test1 = model1.predict(Xp=X_test, samples=SAMPLES)
mean_test2, varv2, var_lower_test2, var_upper_test2, cov_mat_test2 = model2.predict(Xp=X_test, samples=SAMPLES)

def RMSE(pred, test):
    rmse = torch.mean(torch.pow(pred - test, 2)).sqrt()
    return rmse

print("RMSE x_pred vs. y1_test:", RMSE(mean_test1, y1_test).item())
print("RMSE x_pred vs. y2_test:", RMSE(mean_test2, y2_test).item())

def plot_error(ax, x, pred, test):
    for i in range(len(pred)):
        if i == 0:
            ax.plot3D([x[i,0], x[i,0]], [x[i,1], x[i,1]], [pred[i], test[i]],
                c='r', linestyle='dotted', lw='1.5', alpha=1, label="Error")
        else:
            ax.plot3D([x[i,0], x[i,0]], [x[i,1], x[i,1]], [pred[i], test[i]],
                c='r', linestyle='dotted', lw='1.5', alpha=1)



""" PLOT PREDICTIONS """
# PREPARE MESHES FOR PLOTTING
mesh_pred_mean1 = get_mesh_y(tensor=mean1, f_num=PRED_F_STEPS, t_num=PRED_T_STEPS)
mesh_pred_lower1 = get_mesh_y(tensor=var_lower1, f_num=PRED_F_STEPS, t_num=PRED_T_STEPS)
mesh_pred_upper1 = get_mesh_y(tensor=var_upper1, f_num=PRED_F_STEPS, t_num=PRED_T_STEPS)

mesh_pred_mean2 = get_mesh_y(tensor=mean2, f_num=PRED_F_STEPS, t_num=PRED_T_STEPS)
mesh_pred_lower2 = get_mesh_y(tensor=var_lower2, f_num=PRED_F_STEPS, t_num=PRED_T_STEPS)
mesh_pred_upper2 = get_mesh_y(tensor=var_upper2, f_num=PRED_F_STEPS, t_num=PRED_T_STEPS)

# CREATE FIGURE
matplotlib.rcParams.update({'font.size': 24})
fig_pred = plt.figure(figsize=(20, 8))
plt.suptitle(DATA_SET)

# PLOT DATA FOR 1
ax_pred1 = fig_pred.add_subplot(1, 2, 1, projection="3d")
ax_pred1.set_title("Real part")
ax_pred1.plot_surface(mesh_fp, mesh_tp, mesh_pred_mean1,
                 cmap="jet", alpha=0.35)
ax_pred1.plot_wireframe(mesh_f, mesh_t, mesh_y1,
            rstride=1, cstride=0, color='k', linestyle='-', alpha=0.35)
ax_pred1.scatter(mesh_f_train, mesh_t_train, mesh_y1_train,
            marker='o', label="Training data", s=marksize, facecolor=(0,0,0,0), edgecolor=(0,0,0,0.65), linewidth=2)
ax_pred1.scatter(mesh_f_test, mesh_t_test, mesh_y1_test,
            color='k', marker='*', label="Test data", alpha=0.35, s=marksize)

plot_error(ax_pred1, X_test, mean_test1, y1_test)

# PLOT INFO
ax_pred1.set_xlabel("$f \ [kHz]$")
ax_pred1.set_ylabel("$t \ [s]$")
ax_pred1.set_zlabel("$\mathcal{Re} \ (H(f,t))$")
FREQ_SKIP = 12
freqs = data.get_frequencies()[0::FREQ_SKIP]
freqs_line = np.linspace(-1, 1, data.get_f_num())[0::FREQ_SKIP]
ax_pred1.set_xticks(freqs_line)
ax_pred1.set_xticklabels(freqs)
ax_pred1.set_yticks(data.get_time())
ax_pred1.set_yticklabels(np.round((data.get_time()+1)/2 * data.get_last_time(), 3))

# PLOT DATA FOR 2
ax_pred2 = fig_pred.add_subplot(1, 2, 2, projection="3d")
ax_pred2.set_title("Imaginary part")
ax_pred2.plot_surface(mesh_fp, mesh_tp, mesh_pred_mean2,
                 cmap="jet", alpha=0.35)
ax_pred2.plot_wireframe(mesh_f, mesh_t, mesh_y2,
            rstride=1, cstride=0, color='k', linestyle='-', alpha=0.35)
ax_pred2.scatter(mesh_f_train, mesh_t_train, mesh_y2_train,
            marker='o', label="Training data", s=marksize, facecolor=(0,0,0,0), edgecolor=(0,0,0,0.65), linewidth=2)
ax_pred2.scatter(mesh_f_test, mesh_t_test, mesh_y2_test,
            color='k', marker='*', label="Test data", alpha=0.35, s=marksize)
plot_error(ax_pred2, X_test, mean_test2, y2_test)

# PLOT INFO
ax_pred2.set_xlabel("$f \ [kHz]$")
ax_pred2.set_ylabel("$t \ [s]$")
ax_pred2.set_zlabel("$\mathcal{Im} \ (H(f,t))$")
# FREQ_SKIP = 12
# freqs = data.get_frequencies()[0::FREQ_SKIP]
# freqs_line = np.linspace(-1, 1, data.get_f_num())[0::FREQ_SKIP]
# ax_pred2.set_xticks(freqs_line)
# ax_pred2.set_xticklabels(freqs)
# ax_pred2.set_yticks(data.get_time())
# ax_pred2.set_yticklabels(np.round((data.get_time()+1)/2 * data.get_last_time(), 3))

fig_pred.tight_layout(pad=3)
ax_pred1.xaxis.labelpad=30
ax_pred1.yaxis.labelpad=30
ax_pred1.zaxis.labelpad=30
ax_pred2.xaxis.labelpad=30
ax_pred2.yaxis.labelpad=30
ax_pred2.zaxis.labelpad=30

freqs = data.get_frequencies()
ax_pred2.set_xticks([-1, 0, 1])
ax_pred2.set_xticklabels([str(freqs[0]), "0", str(freqs[-1])])
time_str = str(np.around((data.get_time()[-1]+1)/2 * data.get_last_time(), 3))
ax_pred2.set_yticks([data.get_time()[-0], data.get_time()[-1]])
ax_pred2.set_yticklabels(["0", time_str])
ax_pred2.set_zlim([-1, 1])
ax_pred2.set_zticks([-1, 0, 1])
ax_pred2.set_zticklabels(["-1", "0", "1"])

ax_pred1.set_xticks([-1, 0, 1])
ax_pred1.set_xticklabels([str(freqs[0]), "0", str(freqs[-1])])
ax_pred1.set_yticks([data.get_time()[-0], data.get_time()[-1]])
ax_pred1.set_yticklabels(["0", time_str])
ax_pred1.set_zlim([0, 2])
ax_pred1.set_zticks([0, 1, 2])
ax_pred1.set_zticklabels(["0", "1", "2"])

ax_pred1.legend(loc='lower center', bbox_to_anchor=(0.5, -0.18),
    fancybox=True, shadow=False, ncol=3)
ax_pred2.legend(loc='lower center', bbox_to_anchor=(0.5, -0.18),
    fancybox=True, shadow=False, ncol=3)

# ax_pred1.set_zticks([-1, 0, 1])
# ax_pred1.set_zticklabels(["-1","0", "1"])
#
# ax_pred2.set_zticks([-1, 0, 1])
# ax_pred2.set_zticklabels(["-1","0", "1"])


# """ PLOT COVARIANCE MATRIX """
# fig_cov = plt.figure(figsize=(4, 8))
# plt.suptitle("Posterior covariance matrix")
# cmap = 'jet'
#
# ax_cov1 = fig_cov.add_subplot(2, 1, 1)
# ax_cov1.set_title("Real part")
# ax_cov1.pcolormesh(cov_mat1, cmap=cmap)
# plt.gca().invert_yaxis()
# plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[]);
#
# ax_cov2 = fig_cov.add_subplot(2, 1, 2)
# ax_cov2.set_title("Imaginary part")
# ax_cov2.pcolormesh(cov_mat2, cmap=cmap)
# plt.gca().invert_yaxis()
# plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[]);



""" PLOT LOSS """
matplotlib.rcParams.update({'font.size': 16})
fig_loss = plt.figure(figsize=(6,3)) # create figure
plt.suptitle("Loss")
ax_loss = fig_loss.add_subplot() # create subplot
ax_loss.plot(lls1, # plot points
    color='k', label="Real")
ax_loss.plot(lls2, # plot points
    color='k', label="Imaginary", linestyle='--')
ax_loss.set_xlabel("Iterations")



""" PLOT FIGURES """
ax_loss.legend()
# fig_data.tight_layout(pad=3.0)
# fig_pred.tight_layout(pad=3.0)
# fig_loss.tight_layout(pad=3.0)
plt.show()




""" RESULTS 2020-04-06
TEST=ALL TRAIN=ALL
RMSE x_pred vs. y1_test: 0.04114291071891785
RMSE x_pred vs. y2_test: 0.04577477648854256

TEST=1 TRAIN=1
RMSE x_pred vs. y1_test: 0.09019124507904053
RMSE x_pred vs. y2_test: 0.0535380020737648

TEST=2 TRAIN=1
RMSE x_pred vs. y1_test: 0.06880462169647217
RMSE x_pred vs. y2_test: 0.05571197345852852

TEST=3 TRAIN=1
RMSE x_pred vs. y1_test: 0.07794246822595596
RMSE x_pred vs. y2_test: 0.06084160879254341

TEST=4 TRAIN=1
RMSE x_pred vs. y1_test: 0.08086462318897247
RMSE x_pred vs. y2_test: 0.0675506740808487

TEST=6 TRAIN=1
RMSE x_pred vs. y1_test: 0.09180738031864166
RMSE x_pred vs. y2_test: 0.21892714500427246

TEST=8 TRAIN=1
RMSE x_pred vs. y1_test: 0.11572924256324768
RMSE x_pred vs. y2_test: 0.20939308404922485

TEST=10 TRAIN=1
RMSE x_pred vs. y1_test: 0.09954195469617844
RMSE x_pred vs. y2_test: 0.2484954595565796

TEST=12 TRAIN=1
RMSE x_pred vs. y1_test: 0.20513486862182617
RMSE x_pred vs. y2_test: 0.3005870282649994


"""