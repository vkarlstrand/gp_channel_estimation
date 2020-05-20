""" gpr1d1_cb_autocorr.py

    Gaussian Process Regression (GPR)
    * One GP.
    * One dimensional input:    Frequency (f).
    * One dimensional output:   Absolute value of real (a) and imaginary part (b).


    The coherence bandwidth is evaluated by computing autocorrelation function.

    Setup: Python 3.6.9, GpPyTorch v1.0.0

"""



""" IMPORTS
    Import all neccessary libraries, classes and functions.
"""
from Data import Data # parser to import wifi data from csv-file
from Data import get_mesh_x, get_mesh_y, plot3D

import numpy
import scipy
from scipy.signal import find_peaks
from scipy import interpolate
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as mcolors

import torch
import gpytorch
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.distributions import MultivariateNormal
from torch.optim import Adam
from gpytorch.mlls import ExactMarginalLogLikelihood



""" SET UP GP MODEL
"""
class GP(gpytorch.models.ExactGP):
    def __init__(self, x_train, y_train, likelihood):
        super(GP, self).__init__(x_train, y_train, likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    def optimize(self, x, y, likelihood, epochs, lr):
        self.train()
        likelihood.train()
        optimizer = Adam([{'params': self.parameters()}], lr=lr)
        mll = ExactMarginalLogLikelihood(likelihood, self)
        for i in range(epochs):
            optimizer.zero_grad()
            output = self(x)
            loss = -mll(output, y)
            loss.backward()
            ls = self.covar_module.base_kernel.lengthscale.item()
            ns = self.likelihood.noise.item()
            #print(ls, flush=True)
            print(".", end="", flush=True)
            optimizer.step()
        print(".", flush=True)
        return loss, ls, ns

    def predict(self, xpred, likelihood):
        self.eval()
        likelihood.eval()
        with torch.no_grad():
            preds = self(xpred) # preds for function
            #preds = likelihood(self(xpred)) # preds for observations y
            mean = preds.mean
            var = preds.variance
            cov = preds.covariance_matrix
            lower, upper = preds.confidence_region()
        return mean, var, lower, upper, cov




def TransferFunction(X, Y, range_slices, plot=False):
    if plot:
        matplotlib.rcParams.update({'font.size': 14})
        fig = plt.figure(figsize=(9, 4))
        ax1 = fig.add_subplot(1, 1, 1)

    DC = 0
    T_SAMPLES = 52 + 11*DC + DC # bins incl. zero
    Fs = T_SAMPLES * 312.5 * 1000 # Hz
    Fd = 312.5 * 1000 # Hz
    Ts = 1/Fs # s

    F_ABS = numpy.zeros(T_SAMPLES)
    F_REAL = numpy.zeros(T_SAMPLES)
    F_IMAG = numpy.zeros(T_SAMPLES)


    for slice in range_slices:
        t_slice_np = data.get_slice(xy='x', vector=X, time=slice, freq=None)
        f = t_slice_np[:,0]
        Y_slice_np = data.get_slice(xy='y', vector=Y, time=slice, freq=None)
        fd = (Y_slice_np[:,0] + (1j)*Y_slice_np[:,1])

        if DC:
            ls = fd[25]
            rs = fd[27]
            mean = (ls+rs)/2
            #mean = 7 # test
            f = numpy.insert(f, 26, 0)
            fd = numpy.insert(fd, 26, mean)
            f_before = [-31, -30, -29, -28, -27]
            f_after = [27, 28, 29, 30, 31, 32]
            fd_before = [0, 0, 0, 0, 0]
            fd_after = [0, 0, 0, 0, 0, 0]
            f = numpy.insert(f, 0, f_before)
            f = numpy.append(f, f_after)
            fd = numpy.insert(fd, 0, fd_before)
            fd = numpy.append(fd, fd_after)

        fd_abs = numpy.absolute(fd)
        fd_real = numpy.real(fd)
        fd_imag = numpy.imag(fd)
        F_ABS = numpy.vstack((F_ABS, fd_abs))
        F_REAL = numpy.vstack((F_REAL, fd_real))
        F_IMAG = numpy.vstack((F_IMAG, fd_imag))

        # if plot:
        #     ax1.plot(f, fd_abs, c='k', lw=1.5, alpha=0.05)


    F_ABS_MEAN = numpy.mean(F_ABS[1:,], axis=0) # [1:,] so to remove zeros in begining
    F_REAL_MEAN = numpy.mean(F_REAL[1:,], axis=0)
    F_IMAG_MEAN = numpy.mean(F_IMAG[1:,], axis=0)

    F_ABS_MIN = numpy.min(F_ABS[1:,], axis=0)
    F_ABS_MAX = numpy.max(F_ABS[1:,], axis=0)

    F_ABS = F_ABS[1:,]

    if plot:
        matplotlib.rcParams.update({'font.size': 14})
        ax1.set_title(DATA_SET)
        ax1.set_xlabel("$f\ [kHz]$")
        ax1.set_ylabel("$|H(f)|$")
        # ax1.plot(f, F_ABS_MEAN, c='k', lw=1.5, alpha=1, label="Mean")
        for i in range(len(F_ABS_MIN)):
                ax1.plot([f[i], f[i]], [0, F_ABS_MEAN[i]], c='k', alpha=1, lw=0.5)
        ax1.scatter(f, F_ABS_MEAN, 20, marker='o', facecolors='none', edgecolors='k', alpha=1)


        # ax1.plot(f, F_ABS_MAX, c='k', linestyle='--', label="Max", alpha=0.4, lw=0.75)
        # ax1.plot(f, F_ABS_MIN, c='k', linestyle='dotted', label="Min", alpha=0.4, lw=0.75)
        # ax1.fill_between(f, F_ABS_MIN, F_ABS_MAX, color='k', alpha=0.1)
        #
        # ax1.plot(f, F_REAL_MEAN, c='tab:green', ls='--', alpha=0.75, lw=1.5, label="Mean real")
        # ax1.plot(f, F_IMAG_MEAN, c='tab:red',ls='dotted', alpha=0.75, lw=1.5, label="Mean imaginary")


        ax1.plot([f[0], f[-1]], [0, 0], c='k', lw=0.5, alpha=0.5, ls='dotted')
        XTICKS = numpy.asarray([-26, -13, 0, 13, 26]) * Fd/1000 # kHz
        XVALS = [-26, -13, 0, 13, 26]
        ax1.set_xticks(XVALS)
        ax1.set_xticklabels(XTICKS)
        ax1.set_xlim([-26.5, 26.5])
        # ax1.legend(ncol=3, loc='lower center')
        # ax1.grid(which='major')
        ax1.set_ylim([-0.25*numpy.amax(F_ABS_MEAN), 1.25*numpy.amax(F_ABS_MEAN)])

        fig.tight_layout(pad=3)




def AutoCorr(X, Y, kappa, subset):
    M = 53
    N = 10
    N = len(subset)
    s = numpy.zeros(M, dtype=numpy.complex)

    for k in range(0, M):
        for n in range(0, N):
            t_slice_np = data.get_slice(xy='x', vector=X, time=subset[n], freq=None)
            f = t_slice_np[:,0]
            Y_slice_np = data.get_slice(xy='y', vector=Y, time=subset[n], freq=None)
            fd = (Y_slice_np[:,0] + (1j)*Y_slice_np[:,1])
            ls = fd[25]
            rs = fd[27]
            mean = (ls+rs)/2
            f = numpy.insert(f, 26, 0) # insert interpolated zero
            fd = numpy.insert(fd, 26, mean)
            for m in range(0, M-k):
                s[k] += (numpy.conjugate(fd[m]) * fd[m+k])
        s[k] /= 1/(N*(M-k))

    s_abs = numpy.absolute(s)
    s_norm = s_abs/s_abs[0]

    f = data.get_freq()
    f = numpy.insert(f, 26, 0)
    interp = interpolate.interp1d(x=f, y=s_norm, kind='linear')
    TIMES = 8
    NUM = M * TIMES
    fx = numpy.linspace(-26, 26, NUM)
    s_norm = interp(fx)

    k = NUM - 1
    Bc = 0
    K = 0
    while k >= 0:
        if s_norm[k] > kappa:
            Bc = k * 312.5/TIMES
            K = k
            break
        k -= 1

    return Bc, K, s_norm



def Lengthscale_Autocorr(X, Y, kappa, range_slices, plot=False):
    matplotlib.rcParams.update({'font.size': 14})
    fig = plt.figure(figsize=(8, 3.5))
    #fig.suptitle(DATA_SET)
    ax1 = fig.add_subplot(1, 1, 1)
    #ax1.set_title("Normalized autocorrelation of transfer function")
    TIMES = 8
    f = numpy.linspace(-26, 26, 53*TIMES)


    S = numpy.zeros(53*TIMES, dtype=numpy.complex)
    IND = numpy.zeros(1)
    BC = numpy.zeros(1)
    LS = numpy.zeros(1)

    for slice in range_slices:
        window = range(slice-1, slice+2)
        Bc, K, s_norm = AutoCorr(X, Y, kappa, window)
        S = numpy.vstack((S, s_norm))
        IND = numpy.vstack((IND, K))
        BC = numpy.vstack((BC, Bc))

        if plot:
            ax1.plot(f, s_norm, c='k', lw=1, alpha=0.05)

        t_slice_np = data.get_slice(xy='x', vector=X, time=slice, freq=None)
        freq = t_slice_np[:,0]
        Y_slice_np = data.get_slice(xy='y', vector=Y, time=slice, freq=None)
        fd = (Y_slice_np[:,0] + (1j)*Y_slice_np[:,1])
        abs = numpy.absolute(fd)
        x = freq
        y = abs

        f_train = torch.from_numpy(x).reshape(-1,1).type('torch.FloatTensor').contiguous()
        abs_train = torch.from_numpy(y).reshape(-1,).type('torch.FloatTensor').contiguous()

        EPOCHS = 1500 # number of iterations during training
        LR = 0.025 # learning rate during training

        gpytorch.settings.skip_posterior_variances(state=False)
        gpytorch.settings.lazily_evaluate_kernels(state=False)
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = GP(f_train, abs_train, likelihood)
        loss, ls, ns = model.optimize(f_train, abs_train, likelihood, EPOCHS, LR)
        LS = numpy.vstack((LS, ls))

        print("ls:", ls)
        print("Bc:", Bc)

    S_MEAN = numpy.mean(S[1:,], axis=0)
    S_MAX = numpy.mean(S[1:,], axis=0)
    S_MIN = numpy.mean(S[1:,], axis=0)
    IND_MEAN = numpy.mean(IND[1:,], axis=0)
    BC_MEAN = numpy.mean(BC[1:,], axis=0)
    Bc_str = "$B_c|_{\\kappa=0.75}=$" + str(int(BC_MEAN[0])) + " [kHz]"
    Bc_str = "$B_c$"
    kappa_str ="$\kappa$"
    if plot:
        ax1.text(f[int(IND_MEAN)], -0.045, Bc_str, ha='center', va='top')
        ax1.scatter(f[int(IND_MEAN)], S_MEAN[int(IND_MEAN)], c='k', marker='o')
        #ax1.scatter(f[int(IND_MEAN)], 0, c='k', marker='x')
        ax1.plot(f, S_MEAN, c='k', lw=1.5, alpha=1, label="Mean")
        ax1.plot([-26, 26], [0, 0], c='k', lw=1, alpha=1)
        ax1.plot([f[int(IND_MEAN)], f[int(IND_MEAN)]], [0, S_MEAN[int(IND_MEAN)]], lw=1, ls='dotted', c='k')
        ax1.plot([-26, f[int(IND_MEAN)]], [S_MEAN[int(IND_MEAN)], S_MEAN[int(IND_MEAN)]], lw=1, ls='dotted', c='k')
        ax1.plot([f[int(IND_MEAN)], f[int(IND_MEAN)]+5], [S_MEAN[int(IND_MEAN)], S_MEAN[int(IND_MEAN)]], lw=1, ls='dotted', c='k')
        ax1.text(f[int(IND_MEAN)]+5, S_MEAN[int(IND_MEAN)], kappa_str, ha='left', va='center')
        # ax1.legend()
        ax1.set_xlim([-26, 26])
        ax1.set_ylim([0, 1.05])
        ax1.set_xlabel("$k \cdot \delta f$ [kHz]")
        ax1.set_ylabel("$|s [k]|_{norm}$")
        XTICKS = numpy.asarray([-26, -13, 0, 13, 26])
        XTICKLABELS = (26+XTICKS) * 312.5
        ax1.set_xticks(XTICKS)
        ax1.set_xticklabels(XTICKLABELS)
        fig.tight_layout(pad=2)

    BC = BC[1:,]
    LS = LS[1:,]

    return BC, LS







""" LOAD DATA
    Initalize data. Load data. Prepare data and put into vectors.

    Version 1: Version 1: LIMIT=102 JUMP=0 SKIP=500 SKIP_CLOSE=0.25 MAC_ID="8c:85:90:a0:2a:6f"
"""
FILEs = ["data/wifi_capture_190331.csv" , "data/wifi_capture_190413.csv",
         "data/wifi_capture_20190329.csv", "data/wifi_capture_20190331_in2outdoor.csv",
         "data/wifi_capture_20190412.csv"]
MAC_IDs = ["8c:85:90:a0:2a:6f", "1c:b7:2c:d7:67:ec", "ff:ff:ff:ff:ff:ff",
           "34:a3:95:c3:57:10", "a0:93:51:72:24:0f"]
LIMIT = 102 # number of time samples to import
JUMP = 0 # number of time samples to jump over after each imported time sample
SKIP = 0 # number of time samples to skip over in the beginning
SKIP_CLOSE = 0 # if time sample is within SKIP_CLOSE to previous time sample imported, skip it
FILE = FILEs[0] # file to import from 190331 190413 190329 190412
MAC_ID = MAC_IDs[3]

# DATA SETS
# D1: 03 - JUMP=0 SKIP=0 SKIP_CLOSE=0
# D2: 12 - JUMP=0 SKIP=500 SKIP_CLOSE=0
# D3: 13 - JUMP0 SKIP=0 SKIP_CLOSE=0
# D4: 20 - JUMP0 SKIP=0 SKIP_CLOSE=0
# D5: 41 - JUMP0 SKIP=1000 SKIP_CLOSE=0

# D6  33
# D7  22
# D8  43

data = Data(name="Coherence bandwidth") # initalie data object
data.load(csv_file=FILE, mac_str=MAC_ID, limit=LIMIT, skip=SKIP,
          jump=JUMP, skip_close=SKIP_CLOSE) # load data
data.prep() # prepare data and put into vectors
data.norm_effect() # normalize so energy is consistent
# plot3D(data, "Before phase compensation")
# data.norm_phase()
# plot3D(data, "After phase compensation")

X = data.get_x() # get imported input points
Y = data.get_y() # get imported output points


""" RUN """
RANGE_SLICES = range(6,56)
DATA_SET = "Data set 1"

# Plot one GPR.
#LengthScaleGP(X,Y, RANGE_SLICES, plot=True)

kappa = 0.75
PLOT = True

TransferFunction(X=X, Y=Y, range_slices=RANGE_SLICES, plot=True)

BC, LS = Lengthscale_Autocorr(X, Y, kappa, RANGE_SLICES, plot=PLOT)
print("BC:", BC)
print("LS:", LS)
print("BC/LS:", BC/LS)





plt.show()
