""" gpr1d1_predict_freq.py

    Gaussian Process Regression (GPR)
    * Two GPs for each output.
    * One dimensional input:    Frequency (f).
    * One dimensional output:   Real part (a) and imaginary part (b).
    * Assumptions:
        -   Real and imaginary part are considered independent.

    Description:
    UsesDGP to estimate transfer functions of 1 time-slices with different
    proportions of training data and test data. This to try and evaluate how many
    training data points are needed to approximate the transfer function for a certain
    time sample.

    This is done for many slices and an average is taken.

    Setup: Python 3.6.9, GpPyTorch v1.0.0

"""



""" IMPORTS """
import numpy
import matplotlib.pyplot as plt
import torch
import gpytorch

from Data import Data # for importing wifi data, own class
from Data import get_mesh_x, get_mesh_y
from mpl_toolkits.mplot3d import Axes3D
import matplotlib



""" SET UP GP MODEL
"""
class GP(gpytorch.models.ExactGP):
    def __init__(self, x_train, y_train, likelihood):
        """ Takes training data and likelihood and constructs objects neccessary
        for 'forward' module. Commonly mean and kernel module. """
        super(GP, self).__init__(x_train, y_train, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        prior_ls = gpytorch.priors.NormalPrior(3, 3)
        prior_os = gpytorch.priors.NormalPrior(4, 3)
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(lengthscale_prior=prior_ls),
            outputscale_prior=prior_os)


    def forward(self, x):
        """ Takes some NxD data and returns 'MultivariateNormal' with prior mean
        and covariance, i.e. mean(x) and NxN covariance matrix K_xx. """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def optimize(self, x, y, epochs, lr):
        gpytorch.settings.skip_posterior_variances(state=False)
        gpytorch.settings.lazily_evaluate_kernels(state=False)
        self.train()
        self.likelihood.train()

        optimizer = torch.optim.Adam([{'params': self.parameters()}], lr=lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)

        for i in range(epochs):
            optimizer.zero_grad()
            output = self(x)
            loss = -mll(output, y)
            loss.backward()
            #print('Iter %d/%d - Loss: %.3f' % (i + 1, epochs, loss.item()))
            optimizer.step()

    def predict(self, x, likelihood):
        self.eval()
        likelihood.eval()

        with torch.no_grad():
            preds = self(x)
            mean = preds.mean
            var = preds.variance
            lower, upper = preds.confidence_region()
            cov_mat = preds.covariance_matrix

        return mean, var, lower, upper, cov_mat



""" FUNCTIONS FOR EVALUATIONS
    Compute e.g. RMSE, variance etc. for slices with different proportions of
    training data.
"""

def plot_error(ax, x, pred, test):
    for i in range(len(pred)):
        if i != len(pred)-1:
            ax.plot([x[i], x[i]], [pred[i], test[i]],
                c='r', linestyle='dotted', lw='1', alpha=1)
        else:
            ax.plot([x[i], x[i]], [pred[i], test[i]],
                c='r', linestyle='dotted', lw='1', alpha=1, label="Error")



def gp_freq_slice(slice, plot=False):
    # Get slice in time, i.e. array of freqeuncies over that time slice.
    t_slice_np = data.get_slice(xy='x', vector=X_np, time=slice, freq=None)
    Y_slice_np = data.get_slice(xy='y', vector=Y_np, time=slice, freq=None)
    # Convert real part a and imaginary part b into a 52x1 complex vector.
    Y_slice_complex_np = (Y_slice_np[:,0] + (1j)*Y_slice_np[:,1])
    # Compute absolute value of slice.
    abs = numpy.absolute(Y_slice_complex_np)
    # Take frequency from slices and reskape.
    f = t_slice_np[:,0].reshape(-1,1)

    if plot:
        fig = plt.figure(figsize=(8, 11))
        matplotlib.rcParams.update({'font.size': 14})
        plt.suptitle("RMSE of $\mathcal{D}_{TE}$ depending on proportion of $\mathcal{D}_{TR}$")

    # Initalize variables and result arrays.
    i = 1
    rmse = []
    var = []

    for ind in IND_TRAIN:
        # Take indicices in ind as training data. Convert to tensor.
        f_train = numpy.take(f, ind).reshape(-1,1)
        f_train = torch.from_numpy(f_train).type('torch.FloatTensor').contiguous()
        abs_train = numpy.take(abs, ind)
        abs_train = torch.from_numpy(abs_train).type('torch.FloatTensor').contiguous()
        # Take all except indicies in ind as test data. Convert to tensor.
        f_test = numpy.delete(f, ind).reshape(-1,1)
        f_test = torch.from_numpy(f_test).type('torch.FloatTensor').contiguous()
        abs_test = numpy.delete(abs, ind)
        abs_test = torch.from_numpy(abs_test).type('torch.FloatTensor').contiguous()

        # Initialize GP, train and predict for current training data and slice.
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = GP(f_train, abs_train, likelihood)
        model.optimize(f_train, abs_train, epochs=EPOCHS, lr=LR)
        mean_test, var_test, lower_test, upper_test, cov_mat_test = model.predict(f_test, likelihood)

        # Compute results and store.
        error = torch.mean(torch.pow(abs_test - mean_test, 2)).sqrt()
        rmse.append(error.item())

        var.append(torch.mean(var_test).item())


        if plot:
            pred = torch.linspace(-28, 28, 100).reshape(-1,1).type('torch.FloatTensor').contiguous()
            mean_pred, var_pred, lower_pred, upper_pred, cov_mat_pred = model.predict(pred, likelihood)
            pred = pred.reshape(-1,)

            str_error = "Proportion $k_{TR} =" + "\\frac{" + str(len(ind)) + "}{52}$, RMSE: " + str(numpy.around(error.item(), 5))
            ax = fig.add_subplot(len(IND_TRAIN), 1 , i)
            # Plot Power Delay Profile and 0-line.
            marksize = 40
            ax.scatter(f_train, abs_train,
                label="Training points", alpha=0.65, marker='o', s=marksize, facecolors='none', edgecolors='k')
            ax.scatter(f_test, abs_test,
                label="Test points", marker='*', alpha=0.65, s=marksize, facecolors='none', edgecolors='k')
            ax.plot(pred.numpy(), mean_pred.numpy(),
                color='k', lw=1.5, label="Mean", alpha=1, linestyle='-')
            ax.fill_between(pred.numpy(), lower_pred.numpy(), upper_pred.numpy(),
                alpha=0.15, color='k', label="Variance (95%)")
            ax.text(0, 6, str_error, ha='center')
            plot_error(ax, f_test, abs_test, mean_test)
            #ax.plot([pred[0], pred[-1]], [0, 0], c='k', lw=0.5, alpha=0.5, ls='dotted') # 0-line
            # freqs = data.get_frequencies()
            # freqs_ind = [0, 13, 26, 39, 51]
            # freqs_x = numpy.asarray(freqs_ind) - 25
            # freqs_freq = numpy.take(freqs, freqs_ind)
            ax.set_xticks([])
            ax.set_xticklabels([])
            ax.set_xlim([-26.5, 26.5])
            ax.set_yticks([0, 2.5, 5, 8])
            ax.set_yticklabels([0, 2.5, 5, ''])
            ax.set_xlim([-28, 28])
            if i is 1:
                ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.80),
                    fancybox=True, shadow=False, ncol=3)
            elif i is len(IND_TRAIN):
                # ax.set_xticks(freqs_x)
                # ax.set_xticklabels(freqs_freq)
                ax.set_xlabel("$f \ [kHz]$")
                XTICKS = numpy.asarray([-26, -13, 0, 13, 26]) * 312.5 # kHz
                XVALS = [-26, -13, 0, 13, 26]
                ax.set_xticks(XVALS)
                ax.set_xticklabels(XTICKS)
            if i == 4:
                ax.set_ylabel("$|(H(f)|$")
        i += 1
        print('.', end=' ', flush=True)

    return rmse, var

def eval_many_slices(range_slices):
    print_str = "Starting to compute slice " + str(range_slices[0]) + " to " + str(range_slices[-1])
    print(print_str)
    # Initalize result arrays.
    RMSE = numpy.zeros(len(IND_TRAIN))
    VAR = numpy.zeros(len(IND_TRAIN))
    i = 1
    for slice in range_slices:
        slice_str = "Slice " + str(i)
        print(slice_str, end=" " , flush=True)
        rmse, var = gp_freq_slice(slice, plot=False)
        print("")
        RMSE = numpy.vstack((RMSE, rmse))
        VAR = numpy.vstack((VAR, var))
        i = i + 1
    return RMSE, VAR


def many_slices(range_slices, case=True):
    RMSE, VAR = eval_many_slices(range_slices=range_slices)
    RMSE = RMSE[1:,] # remove first row of zeroes
    VAR = VAR[1:,] # remove first row of zeroes
    MEAN_RMSE = numpy.mean(RMSE, axis=0)
    STD_RMSE = numpy.std(RMSE, axis=0)
    MEAN_VAR = numpy.mean(VAR, axis=0)
    STD_VAR = numpy.std(VAR, axis=0)
    print("MEAN_RMSE:", MEAN_RMSE)
    print("STD_RMSE:", STD_RMSE)
    print("MEAN_VAR:", MEAN_RMSE)
    print("STD_VAR:", STD_RMSE)

    matplotlib.rcParams.update({'font.size': 14})
    fig = plt.figure(figsize=(8, 7))
    fig.suptitle(DATA_SET)
    #result_str = "GPR - Mean results for " + str(len(RANGE_SLICES)) + " slices"
    #plt.suptitle(result_str)

    ax1 = fig.add_subplot(2,1,1)
    if case:
        x = numpy.linspace(2, len(IND_TRAIN)+1, len(IND_TRAIN))
        ax1.set_title("Mean RMSE of $\mathcal{D}_{TE}$ / Proportion of $\mathcal{D}_{TR}$")
        ax1.set_xlabel("Proportion of $\mathcal{D}_{TR}$")
        XTICKS = []
        XVALS = []
        for ind in IND_TRAIN:
            XTICKS.append("$\\frac{" + str(len(ind)) + "}{" + str(52) + "}$")
            XVALS.append(len(ind)/52)
        ax1.set_xticks(XVALS)
        ax1.set_xticklabels(XTICKS)
        ax1.tick_params(axis='x', which='major', labelsize=18)
    else:
        x = numpy.linspace(1, len(IND_TRAIN), len(IND_TRAIN))
        ax1.set_title("Mean RMSE of test data vs. training data start position")
        ax1.set_xlabel("Start position of $\mathcal{D}_{TR} $")
        XVALS = numpy.linspace(1, len(IND_TRAIN), len(IND_TRAIN))
        XTICKS = XVALS.astype(dtype=int)
        ax1.set_xticks(XVALS)
        ax1.set_xticklabels(XTICKS)
    ax1.plot(XVALS, MEAN_RMSE,
                     c='k', linestyle='-', lw=2, label="Mean RMSE")
    ax1.scatter(XVALS, MEAN_RMSE,
                     c='k', marker='o')
    ax1.errorbar(XVALS, MEAN_RMSE, STD_RMSE,
                     ecolor='k', marker=None, ls='none', label="Standard deviation")
    # for i in range(len(RMSE)):
    #     ax1.plot(XVALS, RMSE[i],
    #         c='k', linestyle='-', lw=0.5, alpha=0.075)
    shift = -1
    D1 = (numpy.amax(MEAN_RMSE)-numpy.amin(MEAN_RMSE))/2
    for i in range(len(MEAN_RMSE)):
        shift = -1 * shift
        ax1.text(XVALS[i], MEAN_RMSE[i]+D1*shift, str(numpy.around(MEAN_RMSE[i], 3)), ha='center', va='center')
    ax1.set_ylabel("RMSE of $\mathcal{D}_{TE}$")

    ax2 = fig.add_subplot(2,1,2)
    if case:
        ax2.set_title("Mean variance of $\mathcal{D}_{TE}$ / Proportion of $\mathcal{D}_{TR}$")
        ax2.set_xlabel("Proportion of $\mathcal{D}_{TR}$")
        ax2.set_xticks(XVALS)
        ax2.set_xticklabels(XTICKS)
        ax2.tick_params(axis='x', which='major', labelsize=18)
    else:
        ax2.set_title("Mean variance of test data vs. training data start position")
        ax2.set_xlabel("Start position of $\mathcal{D}_{TR} $")
        ax2.set_xticks(XVALS)
        ax2.set_xticklabels(XTICKS)
    ax2.plot(XVALS, MEAN_VAR,
                     c='k', linestyle='-', lw=2, label="Mean variance")
    ax2.scatter(XVALS, MEAN_VAR,
        c='k', marker='o')
    ax2.errorbar(XVALS, MEAN_VAR, STD_VAR,
                     ecolor='k', marker=None, ls='none', label="Standard deviation")
    # for i in range(len(VAR)):
    #     ax2.plot(XVALS, VAR[i],
    #         c='k', linestyle='-', lw=0.5, alpha=0.075)
    shift = -1
    D2 = (numpy.amax(MEAN_VAR)-numpy.amin(MEAN_VAR))/2
    for i in range(len(MEAN_VAR)):
        shift = -1 * shift
        ax2.text(XVALS[i], MEAN_VAR[i]+D2*shift, str(numpy.around(MEAN_VAR[i], 3)), ha='center', va='center')
    ax2.set_ylabel("Variance of $\mathcal{D}_{TE}$")

    ax1.set_ylim([numpy.amin(MEAN_RMSE)-1.3*D1, numpy.max(MEAN_RMSE)+1.5*D1])
    ax2.set_ylim([numpy.amin(MEAN_VAR)-1.3*D2, numpy.max(MEAN_VAR)+1.5*D2])

    ax1.legend()
    ax2.legend()


    fig.tight_layout(pad=2)


def one_slice(slice, plot=True):
    ""
    rmse = gp_freq_slice(slice=slice, plot=plot)
    print(rmse)


def plot_all_slices(range_slices):
    matplotlib.rcParams.update({'font.size': 14})
    fig = plt.figure(figsize=(8, 3))
    for slice in range_slices:
        # Get slice in time, i.e. array of freqeuncies over that time slice.
        t_slice_np = data.get_slice(xy='x', vector=X_np, time=slice, freq=None)
        Y_slice_np = data.get_slice(xy='y', vector=Y_np, time=slice, freq=None)
        # Convert real part a and imaginary part b into a 52x1 complex vector.
        Y_slice_complex_np = (Y_slice_np[:,0] + (1j)*Y_slice_np[:,1])
        # Compute absolute value of slice.
        abs = numpy.absolute(Y_slice_complex_np)
        # Take frequency from slices and reskape.
        f = t_slice_np[:,0].reshape(-1,1)

        ax = fig.add_subplot()
        ax.plot(f, abs, lw=0.5, alpha=0.1, c='k')
        freqs = data.get_frequencies()
        freqs_ind = [0, 13, 26, 39, 51]
        freqs_x = numpy.asarray(freqs_ind) - 25
        freqs_freq = numpy.take(freqs, freqs_ind)
        ax.set_xticks(freqs_x)
        ax.set_xticklabels(freqs_freq)
        ax.set_xlabel("$f \ [kHz]$")
        ax.set_xlim([-26, 26])
        ax.set_ylabel("$|H(f,t_i)|$")
        fig.tight_layout(pad=2)

        print("Last time:", data.get_time()[-1])


""" LOAD DATA
    Initalize data. Load data. Prepare data and put into vectors.
"""
FILEs = ["data/wifi_capture_190331.csv" , "data/wifi_capture_190413.csv",
         "data/wifi_capture_20190329.csv", "data/wifi_capture_20190331_in2outdoor.csv",
         "data/wifi_capture_20190412.csv"]
MAC_IDs = ["8c:85:90:a0:2a:6f", "1c:b7:2c:d7:67:ec", "ff:ff:ff:ff:ff:ff",
           "34:a3:95:c3:57:10", "a0:93:51:72:24:0f"]
LIMIT = 106 # number of time samples to import
JUMP = 0 # number of time samples to jump over after each imported time sample
SKIP = 0 # number of time samples to skip over in the beginning
SKIP_CLOSE = 0 # if time sample is within SKIP_CLOSE to previous time sample imported, skip it
MAC_ALL = False # import all kinds
FILE = FILEs[0] # file to import from
MAC_ID = MAC_IDs[3]
DATA_SET = "Data set 5"
# 1
# 2 LIMIT=206 JUMP=100 SKIP=500

# D1: 03 - JUMP=0 SKIP=0 SKIP_CLOSE=0
# D2: 12 - JUMP=0 SKIP=500 SKIP_CLOSE=0
# D3: 13 - JUMP0 SKIP=0 SKIP_CLOSE=0
# D4: 20 - JUMP0 SKIP=0 SKIP_CLOSE=0
# D5: 41 - JUMP0 SKIP=1000 SKIP_CLOSE=0

data = Data(name="GP") # initalie data object
data.load(csv_file=FILE, mac_str=MAC_ID, limit=LIMIT, skip=SKIP,
          jump=JUMP, skip_close=SKIP_CLOSE, mac_all=MAC_ALL) # load data
data.prep() # prepare data and put into vectors
data.norm_effect()

X_np = data.get_x() # get imported input points
Y_np = data.get_y() # get imported output points
Y_abs_np = data.get_y_abs() # get absolute value, not really used except for plottin in 3D



""" SETTINGS
    Set indicices used as training points, learning rate and epochs.
"""
LR = 0.075
EPOCHS = 750
# Different proportions of training data.
IND_TRAIN = [[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50],
             [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 49],
             [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48],
             [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
             [0, 6, 12, 18, 24, 30, 36, 42, 48],
             [0, 7, 14, 21, 28, 35, 42, 49],
             [0, 8, 16, 24, 32, 40, 48]]

# Carefully chosen for D1.
IND_TRAIN = [[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50],
             [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 49],
             [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 46, 51],
             [2, 7, 11, 16, 21, 25, 30, 35, 40, 46, 50],
             [2, 7, 11, 17, 25, 28, 35, 46, 51],
             [2, 8, 17, 25, 33, 35, 40, 46],
             [1, 8, 16, 25, 35, 40, 46]]

#Different locations of training data for steps of 4.
# IND_TRAIN = [[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48],
#              [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49],
#              [2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50],
#              [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51]]

#Different locations of training data for steps of 8.
# IND_TRAIN = [[0, 8, 16, 24, 32, 40, 48],
#              [1, 9, 17, 25, 33, 41, 49],
#              [2, 10, 18, 26, 34, 42, 50],
#              [3, 11, 19, 27, 35, 43, 51],
#              [4, 12, 20, 28, 36, 44],
#              [5, 13, 21, 29, 37, 45],
#              [6, 14, 22, 30, 38, 46],
#              [7, 15, 23, 31, 39, 47]]

SLICE = 5
RANGE_SLICES = range(1, 51)


""" RUN
    Run programs.
"""
one_slice(slice=SLICE)
#many_slices(range_slices=RANGE_SLICES, case=True) # case=True is for proportion, False for starting location
#plot_all_slices(range_slices=RANGE_SLICES)
plt.show()


""" RESULTS 2020-04-01
    03 Data 1
    MEAN_RMSE:  [0.08901529 0.12368426 0.16377181 0.33398329 0.67409449 0.75782626 0.58812911]
    STD_RMSE:   [0.01665543 0.02229772 0.02956568 0.04422386 0.06218473 0.03802946 0.09808467]
    MEAN_VAR:   [0.08901529 0.12368426 0.16377181 0.33398329 0.67409449 0.75782626 0.58812911]
    STD_VAR:    [0.01665543 0.02229772 0.02956568 0.04422386 0.06218473 0.03802946 0.09808467]

    12 Data 2
    MEAN_RMSE:  [0.00604986 0.00665297 0.00744879 0.00713615 0.00812446 0.00748748 0.0087346 ]
    STD_RMSE:   [0.00098687 0.00109596 0.00125096 0.00105687 0.00110634 0.00132262 0.00166319]
    MEAN_VAR:   [0.00604986 0.00665297 0.00744879 0.00713615 0.00812446 0.00748748 0.0087346 ]
    STD_VAR:    [0.00098687 0.00109596 0.00125096 0.00105687 0.00110634 0.00132262 0.00166319]

    13 Data 3
    MEAN_RMSE:  [0.00759642 0.01374085 0.01916136 0.05501259 0.08377525 0.08862352 0.07386446]
    STD_RMSE:   [0.00101152 0.00223463 0.00252226 0.01908585 0.00522857 0.00577584 0.00502182]
    MEAN_VAR:   [0.00759642 0.01374085 0.01916136 0.05501259 0.08377525 0.08862352 0.07386446]
    STD_VAR:    [0.00101152 0.00223463 0.00252226 0.01908585 0.00522857 0.00577584 0.00502182]

    20 Data 4
    MEAN_RMSE:  [0.01118782 0.01567897 0.02324003 0.01526839 0.02584137 0.03851447 0.05329787]
    STD_RMSE:   [0.00180354 0.00231128 0.00255439 0.00168093 0.00179702 0.00656797 0.00414656]
    MEAN_VAR:   [0.01118782 0.01567897 0.02324003 0.01526839 0.02584137 0.03851447 0.05329787]
    STD_VAR:    [0.00180354 0.00231128 0.00255439 0.00168093 0.00179702 0.00656797 0.00414656]

    41 Data 5
    MEAN_RMSE:  [0.00736925 0.00867288 0.0130441  0.01831799 0.01953822 0.01726607 0.01977246]
    STD_RMSE:   [0.00127546 0.00103605 0.00469764 0.00379437 0.00349932 0.00291825 0.00255378]
    MEAN_VAR:   [0.00736925 0.00867288 0.0130441  0.01831799 0.01953822 0.01726607 0.01977246]
    STD_VAR:    [0.00127546 0.00103605 0.00469764 0.00379437 0.00349932 0.00291825 0.00255378]

"""

""" RESULTS 2020-04-05
    03 Data 1
    MEAN_RMSE: [0.08997711 0.127899   0.16448897 0.31444594 0.65776768 0.74399877
 0.57737561]
STD_RMSE: [0.01636146 0.0332576  0.03250727 0.03815689 0.06937286 0.041251
 0.11267206]
MEAN_VAR: [0.08997711 0.127899   0.16448897 0.31444594 0.65776768 0.74399877
 0.57737561]
STD_VAR: [0.01636146 0.0332576  0.03250727 0.03815689 0.06937286 0.041251
 0.11267206]

    12 Data 2
    MEAN_RMSE: [0.00603565 0.00666113 0.00741316 0.00711243 0.00802555 0.00728731
 0.00837631]
STD_RMSE: [0.0009913  0.00108991 0.00124851 0.00104842 0.00104861 0.00125499
 0.00145159]
MEAN_VAR: [0.00603565 0.00666113 0.00741316 0.00711243 0.00802555 0.00728731
 0.00837631]
STD_VAR: [0.0009913  0.00108991 0.00124851 0.00104842 0.00104861 0.00125499
 0.00145159]

    13 Data 3
    MEAN_RMSE: [0.00754102 0.0135565  0.01930885 0.05291332 0.09161488 0.09312116
 0.08156688]
STD_RMSE: [0.00108262 0.0020895  0.00250247 0.02544127 0.0051786  0.00566107
 0.00489069]
MEAN_VAR: [0.00754102 0.0135565  0.01930885 0.05291332 0.09161488 0.09312116
 0.08156688]
STD_VAR: [0.00108262 0.0020895  0.00250247 0.02544127 0.0051786  0.00566107
 0.00489069]


    20 Data 4
    MEAN_RMSE: [0.01077985 0.01525155 0.02339678 0.01468472 0.0248759  0.02966283
 0.04445426]
STD_RMSE: [0.00207666 0.00267889 0.00335971 0.00168353 0.00257644 0.00916993
 0.00934407]
MEAN_VAR: [0.01077985 0.01525155 0.02339678 0.01468472 0.0248759  0.02966283
 0.04445426]
STD_VAR: [0.00207666 0.00267889 0.00335971 0.00168353 0.00257644 0.00916993
 0.00934407]

    41 Data 5
    MEAN_RMSE: [0.00763383 0.00882437 0.01362774 0.01729052 0.01816586 0.01695255
 0.01815584]
STD_RMSE: [0.00152657 0.00148995 0.0060009  0.00729273 0.00687129 0.00645396
 0.00622514]
MEAN_VAR: [0.00763383 0.00882437 0.01362774 0.01729052 0.01816586 0.01695255
 0.01815584]
STD_VAR: [0.00152657 0.00148995 0.0060009  0.00729273 0.00687129 0.00645396
 0.00622514]


"""
RMSE_MEAN = [[0.08901529, 0.12368426, 0.16377181, 0.33398329, 0.67409449, 0.75782626, 0.58812911],
        [0.00604986, 0.00665297, 0.00744879, 0.00713615, 0.00812446, 0.00748748, 0.0087346],
        [0.00759642, 0.01374085, 0.01916136, 0.05501259, 0.08377525, 0.08862352, 0.07386446],
        [0.01118782, 0.01567897, 0.02324003, 0.01526839, 0.02584137, 0.03851447, 0.05329787],
        [0.00736925, 0.00867288, 0.0130441,  0.01831799, 0.01953822, 0.01726607, 0.01977246]]

RMSE_STD = [[0.01665543, 0.02229772, 0.02956568, 0.04422386, 0.06218473, 0.03802946, 0.09808467],
            [0.00098687, 0.00109596, 0.00125096, 0.00105687, 0.00110634, 0.00132262, 0.00166319],
            [0.00101152, 0.00223463, 0.00252226, 0.01908585, 0.00522857, 0.00577584, 0.00502182],
            [0.00180354, 0.00231128, 0.00255439, 0.00168093, 0.00179702, 0.00656797, 0.00414656],
            [0.00127546, 0.00103605, 0.00469764, 0.00379437, 0.00349932, 0.00291825, 0.00255378]]

RMSE_MEAN = numpy.asarray(RMSE_MEAN)
RMSE_STD = numpy.asarray(RMSE_STD)


def plot_results(mean, std):
    matplotlib.rcParams.update({'font.size': 14})
    fig = plt.figure(figsize=(8, 12))
    TOT = len(mean[0])
    XTICKS = []
    XVALS = []
    X = numpy.asarray([26, 17, 13, 11, 9, 8, 7])
    for x in X:
        XTICKS.append("$\\frac{" + str(x) + "}{" + str(52) + "}$")
        XVALS.append(x/52)
    for i in range(len(mean)):
        ax = fig.add_subplot(len(mean), 1, i+1)
        ax.set_title("Data set " + str(i+1))
        ax.plot(XVALS, mean[i],
                         c='k', linestyle='-', lw=2, label="Mean RMSE")
        ax.scatter(XVALS, mean[i],
                         c='k', marker='o')
        ax.errorbar(XVALS, mean[i], std[i],
                         ecolor='r', marker=None, ls='none', label="Std")
        shift = -1
        D = (numpy.amax(mean[i])-numpy.amin(mean[i]))/2
        for j in range(len(mean[0])):
            shift = -1 * shift
            ax.text(XVALS[j], mean[i][j]+D*shift, str(numpy.around(mean[i][j], 3)), ha='center', va='center')
        ax.set_xticks(XVALS)
        ax.set_xticklabels(XTICKS)
        ax.tick_params(axis='x', which='major', labelsize=18)
        ax.set_xlabel("Proportion of $\mathcal{D}_{TR}$")
        ax.set_ylabel("$log$ RMSE of $\mathcal{D}_{TE}$")
        ax.legend()
        plt.yscale('log')
        plt.grid(which='minor', alpha=0.4)
    fig.tight_layout(pad=2)

#plot_results(RMSE_MEAN, RMSE_STD)
#plt.show()
""" RESULTS 2020-03-26

    100 slices, training data proportion.
    MEAN_RMSE: [0.10556971 0.18990346 0.20897087 0.44657982 0.81551902 0.89213536 0.9017048 ]
    STD_RMSE: [0.02026515 0.04635647 0.04181532 0.10339564 0.15264819 0.11531501 0.18796278]
    MEAN_VAR: [0.10556971 0.18990346 0.20897087 0.44657982 0.81551902 0.89213536 0.9017048 ]
    STD_VAR: [0.02026515 0.04635647 0.04181532 0.10339564 0.15264819 0.11531501 0.18796278]

    100 slices, training data start position.
    MEAN_RMSE: [0.20897087 0.20522473 0.25820281 0.32072124]
    STD_RMSE: [0.04181532 0.04289672 0.03864729 0.05460579]
    MEAN_VAR: [0.20897087 0.20522473 0.25820281 0.32072124]
    STD_VAR: [0.04181532 0.04289672 0.03864729 0.05460579]


"""



""" RESULTS OLD
    100 slices different proportions:
    MEAN_RMSE: [0.10468631 0.18853841 0.20653683 0.43975929 0.80719178 0.88328515 0.89287778]
    STD_RMSE: [0.02012013 0.04616903 0.0412926  0.10195197 0.15137877 0.11425579 0.18670864]
    MEAN_VAR: [0.10468631 0.18853841 0.20653683 0.43975929 0.80719178 0.88328515 0.89287778]
    STD_VAR: [0.02012013 0.04616903 0.0412926  0.10195197 0.15137877 0.11425579 0.18670864]

    200 slices different transfer functions and different proportions:
    MEAN_RMSE: [0.12943458 0.1209787  0.13124762 0.15025405]
    STD_RMSE: [0.05792898 0.0531949  0.06013142 0.07230779]
    MEAN_VAR: [0.12943458 0.1209787  0.13124762 0.15025405]
    STD_VAR: [0.05792898 0.0531949  0.06013142 0.07230779]



    100 slices starting position:
    MEAN_RMSE: [0.20653683 0.20296532 0.25593862 0.31924064]
    STD_RMSE: [0.0412926  0.04231294 0.03838767 0.05481308]
    MEAN_VAR: [0.20653683 0.20296532 0.25593862 0.31924064]
    STD_VAR: [0.0412926  0.04231294 0.03838767 0.05481308]

    200 slices different transfer functions starting positions:
MEAN_RMSE: [0.09310731 0.11209247 0.12943458 0.17676431 0.24759966 0.25553812
 0.25016598]
STD_RMSE: [0.0529049  0.0518329  0.05792898 0.09994436 0.17408952 0.18144622
 0.15435864]
MEAN_VAR: [0.09310731 0.11209247 0.12943458 0.17676431 0.24759966 0.25553812
 0.25016598]
STD_VAR: [0.0529049  0.0518329  0.05792898 0.09994436 0.17408952 0.18144622
 0.15435864]

"""



