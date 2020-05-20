""" IMPORTS """
import numpy as np
import matplotlib.pyplot as plt
import torch
import gpytorch


""" GENERATE DATA """
# NORMALIZE
# Normalize to (-1,1) so variational inference can be done with 'randn' with
# inducing points drawn from N(0,1).
def normalize(x):
    min = torch.min(x)
    x -= min
    max = torch.max(x)
    x = 2*x/max -1
    return x

# CREATE TRAINING DATA
x = torch.linspace(0, 1, 52)  # even
#x = torch.rand(52)            # uniform distrubuted

y = 15*x*torch.sin(15*x)*torch.exp(-5*torch.abs(x))
y1 = y[0:27]
y2 = y[27:35] * 0 + 0.6
y3 = -y[35:37] * 0 - 0.75
y4 = y[37:53] - 0.45
y = torch.cat((y1, y2, y3, y4), dim=0)

y1 = y[0:15]
y2 = y[15:26] * 0 - 1
y3 = y[26:37] * 0 + 0.5
y4 = y[37:52]
y = torch.cat((y1, y2, y3, y4), dim=0)

# y += torch.randn(52)*0.035
print(y.shape)
x = normalize(x)
y = normalize(y)

SKIP = 1 # set to > 1 to skip, e.g. to 2 if to skip every other training data etc.
x = x[0::SKIP]
y = y[0::SKIP]



""" PLOT DATA BEFORE TRAINING """
# Pyplot wants an array of [num_training_points], not [num_training_points, dimension].
# For training, the 'x' will be reshaped further on.
fig = plt.figure(figsize=(10,6)) # create figure
plt.suptitle("Deep Gaussian Process Regression")
ax = fig.add_subplot() # create subplot
ax.scatter(x, y, # plot points
    marker='x', color='k', label="Training points")



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
    # INITALIZE DGP LAYERS
    # Used to create each GP of the DGP's layers.
    def __init__(self, input_dims, output_dims, num_inducing=128, mean_type='constant'):
        # FOR VARIATIONAL INFERENCE: CREATE INDUCING POINTS DRAWN FROM N(0,1)
        if output_dims is None:
            print("num_inducing:", num_inducing)
            print("input_dims:", input_dims)
            inducing_points = torch.randn(num_inducing, input_dims)
            batch_shape = torch.Size([])
        else:
            inducing_points = torch.randn(output_dims, num_inducing, input_dims)
            batch_shape = torch.Size([output_dims])

        # INITALIZE VARIATIONAL DISTRUBUTION
        # The distrubution used for approximation of true posterior distrubution.
        # Cholesky has a full mean vector of size num_induxing and a full covariance
        # matrix of size num_inducing * num_inducing. These are learning during training.
        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=num_inducing,
            batch_shape=batch_shape)

        # INITIALIZE VARIATIONAL STRATEGY
        # Variational strategy wrapper for variational distrubution above.
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True)

        # Call the DeepGPLayer of GPyTorch do initalize the real class for DGPs.
        super(DGPHiddenLayer, self).__init__(variational_strategy, input_dims, output_dims)

        # INITALIZE MEAN
        # The mean module to be used. A true Gaussian is often times constant in it's output.
        if mean_type == 'constant':
            self.mean_module = ConstantMean(batch_shape=batch_shape) # batch_shape so it knows the dimensions
        else: # (if 'linear')
            self.mean_module = LinearMean(input_dims)

        # INITIALIZE KERNEL
        # RBF has no scaling, so wrap it with a ScaleKernel with constant k, that is
        # kernel = k * kernel_rbf. Can make constraints and priors for parameters as well.
        # It's probobly a good idea to set a prior since we normalize the data and have a
        # prior belief about them since we can observe the training data that has a certain appearance.
        # The question is what to set them to. One might have them free to begin with and note
        # what lengthscales turn out good and then constrain to them to get faster convergence
        # for future training.

        #lengthscale_constraint = gpytorch.constraints.Interval(0.0001, 10.0) # needs to be floats
        lengthscale_prior = gpytorch.priors.NormalPrior(0.5, 3.0)
        lengthscale_constraint = None
        #lengthscale_prior = None

        self.covar_module = ScaleKernel(
            RBFKernel(batch_shape=batch_shape, # to set separate lengthscale for each eventuall batch
                      ard_num_dims=input_dims,
                      #active_dims=(0), # set input dims to compute covariance for, tuple of ints corresponding to indices of dimensions
                      lengthscale_constraint=lengthscale_constraint,
                      lengthscale_prior=lengthscale_prior),

                  batch_shape=batch_shape, # for ScaleKernel
                  ard_num_dims=None) # for ScaleKernel

    # INITIALIZE FORWARD
    # Forwards outputs as inputs through the layers.
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class DGP(DeepGP):
    # INITIALIZE DGP MODEL
    # Used to configure the DGP with the GP's initalized above.
    def __init__(self, train_x_shape):
        # Hidden layer, input layer.
        hidden_layer = DGPHiddenLayer(
            input_dims=train_x_shape[-1],
            output_dims=NUM_OUTPUT_DIMS,
            num_inducing=NUM_INDUCING,
            mean_type='linear')

        hidden_layer2 = DGPHiddenLayer(
            input_dims=hidden_layer.output_dims,
            output_dims=NUM_OUTPUT_DIMS,
            num_inducing=NUM_INDUCING,
            mean_type='linear')

        # ... (possibly more hidden layers)

        last_layer = DGPHiddenLayer(
        # Last layer, output layer. Since the output is one dimensional, output_dims
        # is set to None which creates a MultivariateNoraml as output, rather than a
        # MultitaskMultivariateNormal for many dimensions (even though MultitaskMultivariateNormal
        # is considered a wrapper for multidimensional outputs if indepdendent, and not really
        # using the multitask feature and kernel).
            input_dims=hidden_layer.output_dims,
            output_dims=None,
            num_inducing=NUM_INDUCING,
            mean_type='constant') # in general GP's and DGP's should have constant mean if function is considered stationary

        # Call the DeepGP of GPyTorch do initalize the real class for DGPs.
        super().__init__()

        # Set up the model with its different GP's as well as an likelihood.
        # Since it's one dimensional output, GaussianLikelihoodis enough,
        # otherwise MultitaskGaussianLikelihood.
        self.hidden_layer = hidden_layer
        # ... (possibly more hidden layers)
        self.hidden_layer2 = hidden_layer2
        self.last_layer = last_layer
        self.likelihood = GaussianLikelihood()

    # INITIALIZE FORWARD
    # Forwards the outputs as inputs to the next layer.
    # If a deeper DGP is wanted, initalize more hidden layers in DGP __init__ above
    # and put them here in forward().
    def forward(self, inputs):
        hidden_layer_output = self.hidden_layer(inputs)
        # hidden_layer_output_2 = self.hidden_lauer_2(hidden_layer_output_2)
        hidden_layer2_output = self.hidden_layer2(hidden_layer_output)
        output = self.last_layer(hidden_layer2_output)
        return output



""" MODEL AND TRAINING
    Sometimes it seems that the loss is just constant, but with many iterations,
    hopefully it goes done eventuelly quite quickly.
    Sometimes it's just bad, then try and rerun and see if it goes better.
"""
# Reshape training data to [num_training_points, dimensions].
x = x.reshape(-1,1)

# Set parameters for training.
# These may be played with in order to get a convergence and low loss.
EPOCHS = 2000 # number of iterations
LEARNING_RATE = 0.02 # learning rate
SAMPLES = 40 # number of samples to make for approximations, since it's variational and not exakt inference
NUM_INDUCING = 256 # number of inducing points to to use for variational inference
NUM_OUTPUT_DIMS = 1 # (the output/input dimension between hidden layer/last layer)

# Control shapes so the DGP see correct input dimensions.
# x should have shape [num_training_points, dimension]
print("x.shape:", x.shape) # [52, 1]
print("x.shape[0]:", x.shape[0]) # [52] num_training_points
print("x.shape[-1]:", x.shape[-1]) # [1] input dimension
print("x.shape[-2]:", x.shape[-2]) # [52] num_training_points

# Initialize model with the x.shape, requires data on form [num_training_points, dimension]
model = DGP(x.shape)
# Set up the optmizer, Adam is generally the best with adaptive learning rate etc.
opt = torch.optim.Adam([{'params': model.parameters()}], lr=LEARNING_RATE)
# Set up marginal likelihood approximation. Uses the DeepApproximateMLL wrapper.
# The VariationalELBO requires to known the number of num_training_points, i.e. 52.
mll = DeepApproximateMLL(VariationalELBO(model.likelihood, model, x.shape[-2]))

# Set model and likelihood in training mode. For some implementations the 'grad()'
# feature must be turned on and off depending on training or evaluation.
model.train()
model.likelihood.train()

lls = [] # to save loss

# Train by iteration and updating "weights" and parameters.
gpytorch.settings.skip_posterior_variances(state=False) # this is defualt False I think, but just to be sure
for i in range(EPOCHS):
    # Train with certain amount of samples by specificing with SAMPLES, otherwise defualt 10 samples is used.
    with gpytorch.settings.num_likelihood_samples(SAMPLES):
        opt.zero_grad() # reset gradient for each iteration, otherwise the will accumulate
        output = model(x) # returns the resulting MultivariateNormal with current parameters
        loss = -mll(output, y) # evaluate the MultiVariateNormal returned, with training targets y, record loss
        loss.backward() # compute gradient of loss with respect to parameters

        # Prepare info to print:
        hid_ls = model.hidden_layer.covar_module.base_kernel.lengthscale.item()
        out_ls = model.last_layer.covar_module.base_kernel.lengthscale.item()
        noise = model.likelihood.noise.item()

        print('Iter %d/%d:   loss: %.5f   ls 1: %.3f   ls 2: %.3f   likelihood noise 1: %.3f' % (
            #i+1, EPOCHS, loss.item(), 0, 0, 0))
            i+1, EPOCHS, loss.item(), hid_ls, out_ls, noise))

        lls.append(loss.item()) # save loss of current iteration
        opt.step() # update parameters with current gradient descent iteration result



""" PREDICTIONS """
# Set model and likelihood in evaluation mode.
model.eval()
model.likelihood.eval()

# Generate prediction input data to be able to plot mean and variance as a continous function.
xp = torch.linspace(-1.25, 1.25, 200) # use some range close to training, but prehaps a bit outside to see deviation where uncertain
xp = xp.reshape(-1,1) # reshape to [200,1], i.e. 200 predictin points with 1 dimension

# Make predictions with gradients turn off and specificy to use more than the defualt 10 samples.
with torch.no_grad(), gpytorch.settings.num_likelihood_samples(SAMPLES):
    """ CALLING LIKELIHOOD
        1) If likelihood is called with a torch.Tensor object, then it is assumed that
            the input is samples from f(x). This returns the conditional distribution,
            p(y|f(x)).
        2)  If likelihood is called with a MultivariateNormal object, then it is assumed
            that the input is the distribution f(x). This returns the marginal distribution,
            p(y|x).
        Consequently this returns predictions for targets y.

        CALLING MODEL OF DGP
        Upon calling, instead of returning a variational distribution q(f), returns
        samples from the variational distribution. The output is MultivariateNormal if
        output dimension is 1 (None), otherwise MultitaskMultivariateNormal.
        Consequently this returns predictions for the functions.

        CALLING REGULAR GP (NOT USED IN THIS FILE, BUT STILL)
        Calling this model will return the posterior of the latent Gaussian process when
        conditioned on the training data. The output will be a MultivariateNormal.
    """

    # 2) called with MultiVariateNormal that is returned from model(xp) and
    # will calculate the marginal distrubution p(y|x). Returns object with means and variances.
    preds_marg = model.likelihood(model(xp)) # get target predictions
    means_marg = preds_marg.mean # get mean samples from preds_marg object with shape [SAMPLES, num_pred_points]
    vars_marg = preds_marg.variance # get variance samples from preds_marg object with shape [SAMPLES, num_pred_points]
    lowers_marg, uppers_marg = preds_marg.confidence_region() # helper function to get 2 stddev from mean
    # Since DGP is an approximation, samples are returned. For better means and variances, use high SAMPLES.
    # Next, we want to take the mean of the samples to get a "real" mean and variances.
    # This is done along the axis of the SAMPLES with mean(0).
    mean_marg = means_marg.mean(0)
    var_marg = vars_marg.mean(0)
    lower_marg = lowers_marg.mean(0)
    upper_marg = uppers_marg.mean(0)

    # 1) Model called and returns samples from variational distrubution that approximates
    # the true posterior distrubution.
    preds = model(xp) # get function predictions
    means = preds.mean
    mean = means.mean(0)
    vars = preds.variance
    var = vars.mean(0)
    vars_lower, vars_upper = preds.confidence_region()
    var_lower = vars_lower.mean(0)
    var_upper = vars_upper.mean(0)

    cov_mat = preds.covariance_matrix # get covariance matrix of posterior with function predictions
    cov_mat = cov_mat.mean(0) # take mean of all 20 SAMPLES

    f_samples = preds.sample() # get sampled functions, will get 20 as SAMPLES is set as that



""" PLOT PREDICTIONS """
# Plot in the same graph used for plotting traing data.
# First reshape training data to the form pyploy likes.
xp = xp.reshape(-1) # array of size num_pred_points

# Plot mean and 2 stddev. Conert tensors to numpy() for plotting.
ax.plot(xp.numpy(), mean.numpy(),
    label="Mean", linestyle='-', lw='3', color='C0')
ax.fill_between(xp.numpy(), var_lower.numpy(), var_upper.numpy(),
    label="Variance (95%)", color='C0', alpha=0.25)

for i in range(f_samples.shape[-2] - 10): # iterate over sampled functions, -10 since we only want some of them
    ax.plot(xp.numpy(), f_samples[i], # plot each function
        color='C1', alpha=0.1)



""" PLOT COVARIANCE MATRIX """
fig_cov = plt.figure(figsize=(4, 4))
cmap = 'jet' # colormap to use
ax_cov = fig_cov.add_subplot()
ax_cov.set_title("Posterior covariance matrix")
ax_cov.pcolormesh(cov_mat, cmap=cmap)
plt.gca().invert_yaxis()
plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[]);


""" PLOT LOSS """
fig_loss = plt.figure(figsize=(6,3)) # create figure
plt.suptitle("Loss")
ax_loss = fig_loss.add_subplot() # create subplot
ax_loss.plot(lls, # plot points
    color='k')
ax_loss.set_xlabel("Iterations")



""" PLOT FIGURES """
ax.legend() # make sure labels are plotted
fig.tight_layout(pad=3.0) # make some padding so title etc. shows nicely
fig_loss.tight_layout(pad=3.0)
plt.show()
