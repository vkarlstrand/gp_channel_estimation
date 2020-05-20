""" gpr1d1_cb.py

    Gaussian Process Regression (GPR)
    * One GP.
    * One dimensional input:    Frequency (f).
    * One dimensional output:   Absolute value of real (a) and imaginary part (b).
    * Assumptions:
        -   ???

    Description:
    In this file the coherence bandwidth is extimated as the frequency range
    at a certain peak where the covariance is above a certain value, i.e.
    at a certain peak, how far to the left and right can we go before the covariance
    goes below a certain value. This range then becomes the coherence bandwidth for
    that peak.

    The coherence bandwidth is evaluated by:
    1)   Computing delay spread by Discrete Inverse Fourier Transform and Power Delay Profile.
    2)   Finding a relation between coherence bandwidth and length-scale of a RBF.

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



""" FOURIER TRANSFORM FOR SPREAD DELAY
    Compute IFFT of transfer function and get absolute value, real and imaginary part.
    Needs to be "rolled", so that center frequency is in the beginning.
"""
def PowerDelayProfile(h, norm=True):
    """ Takes the absolute value of each element, tau_i, and power of 2, i.e.
        P(tau_i) = |h(tau_i)|^2.
    """
    abs = numpy.absolute(h) # absolute value of each element in h
    pow = numpy.power(abs, 2) # power of 2 of each element in h
    P = pow
    if norm: # if to normalize with respect to noise.
        norm = pow - numpy.median(pow) # compensate for noise
        P = norm
    return P

def MaximumDelay(h, a=0.1, a2=0.1):
    """ Finds the maximum delay T_m where P(tau = T_m) is lower than a set factor dP.
        dP is can be when P is lower than a*P, e.g. 0.1*P.

        However, since P is discrete and has a zero just after one time stamp, the
        total energy is calculated and then when a*tot_energy is reached, that is
        the maximum delay. It's interpreted in the "normal" sense, where T_m is the last
        longest path of significant energy.
    """
    P = PowerDelayProfile(h, norm=True) # get power delay profile of h, compensate for noise with norm=True
    #P = P[0:40] # ignore last taus, defects?
    sum = numpy.sum(P) # get total energy in h
    dP = a * sum # the boundary to find, i.e. when the energi is below a * total energy in h
    # Find T_m, where energy ends.
    for tau in range(len(P)): # iterate over each element in power delay profile
        if sum <= dP: # if the accumulated sum is below or equal to the boundary wanted...
            T_m = tau # ...save tau as T_m...
            break # ...and break.
        else:
            sum -= P[tau] # otherwise keep looking for the boundary
    # Find T_s, where energy starts.
    dP2 = a2*numpy.sum(P)
    sum = 0
    for tau in range(len(P)): # iterate over each element in power delay profile
        if sum >= dP2: # if the accumulated sum is below or equal to the boundary wanted...
            T_s = tau-1 # ...save tau as T_s...
            break # ...and break.
        else:
            sum += P[tau] # otherwise keep looking for the boundary
    return T_s, T_m

def MeanDelay(h):
    """ Find mean delay.
        mu_tau = frac{ sum_{tau_i} tau_i * P(tau_i) }
                     { sum_{tau_i} P(tau)           }

        Since noise is present, only the interval 0 to T_m is considered, i.e.
        we consider only the delays tau_i that have significant energy. The other
        tau_i above T_m are disregarded and considered non-existent.
    """
    P = PowerDelayProfile(h, norm=True)
    T_s, T_m = MaximumDelay(h)
    P = P[T_s:T_m+1] # consider only 0 to T_m, including T_m
    sum = 0
    for tau in range(len(P)):
        sum += tau * P[tau]
    mu_tau = sum/numpy.sum(P)
    return mu_tau

def RMSDelay(h):
    """ Find RMS Delay spread.
        sigma_tau = sqrt( frac { int_{tau=0}^{T_m} (tau - mu_tau)^2 * P(tau) dtau }
                               { int_{tau=0}^{T_m} P(tau) dtau                    })

        As above, only range 0 to and including T_m is considered.
    """
    P = PowerDelayProfile(h)
    mu_tau = MeanDelay(h)
    T_s, T_m = MaximumDelay(h)
    P = P[T_s:T_m+1]
    sum = 0
    for tau in range(len(P)):
        sum += numpy.power((tau - mu_tau), 2) * P[tau]
    div = sum/numpy.sum(P)
    sigma_tau = numpy.sqrt(div)
    return sigma_tau



""" TRANSFER FUNCTION AND IMPULSE RESPONSE FOR RANGE OF SLICES
"""
def TransferFunctionImpulseResponse(X, Y, range_slices, plot=False):
    if plot:
        matplotlib.rcParams.update({'font.size': 14})
        fig = plt.figure(figsize=(9, 5))
        #fig.suptitle(DATA_SET)
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2)

    DC = 1
    T_SAMPLES = 52 + 11*DC + DC # bins incl. zero
    Fs = T_SAMPLES * 312.5 * 1000 # Hz
    Fd = 312.5 * 1000 # Hz
    Ts = 1/Fs # s

    F_ABS = numpy.zeros(T_SAMPLES)
    F_REAL = numpy.zeros(T_SAMPLES)
    F_IMAG = numpy.zeros(T_SAMPLES)
    T_ABS = numpy.zeros(T_SAMPLES)
    T_REAL = numpy.zeros(T_SAMPLES)
    T_IMAG = numpy.zeros(T_SAMPLES)

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

        fd_roll = numpy.roll(fd, shift=T_SAMPLES//2+DC)

        # fd_roll = numpy.roll(fd, shift=52//2+DC)
        # fd_after = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        # f_after = [27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37]
        # fd_roll = numpy.append(fd_roll, fd_after)
        # f = numpy.append(f, f_after)

        fd_roll = fd
        #fd = fd_roll # test
        td = numpy.fft.ifft(fd_roll)
        td_roll = numpy.roll(td, shift=8)
        td = td_roll
        t = (f - numpy.amin(f))
        t = t * Ts * 1000000 # microseconds for samples 0, 1, ..., 52(+1)

        fd_abs = numpy.absolute(fd)
        fd_real = numpy.real(fd)
        fd_imag = numpy.imag(fd)
        F_ABS = numpy.vstack((F_ABS, fd_abs))
        F_REAL = numpy.vstack((F_REAL, fd_real))
        F_IMAG = numpy.vstack((F_IMAG, fd_imag))

        td_abs = numpy.absolute(td)
        td_real = numpy.real(td)
        td_imag = numpy.imag(td)
        T_ABS = numpy.vstack((T_ABS, td_abs))
        T_REAL = numpy.vstack((T_REAL, td_real))
        T_IMAG = numpy.vstack((T_IMAG, td_imag))

        if plot:
            #ax1.plot(f, fd_abs, c='k', lw=0.5, alpha=0.075)
            # ax1.plot(f, fd_real, c='tab:green', lw=0.5, alpha=0.05)
            # ax1.plot(f, fd_imag, c='tab:red', lw=0.5, alpha=0.05)

            ax2.plot(t, td_abs, c='k', lw=0.5, alpha=0.075)
            ax2.plot(t, td_real, c='tab:green', lw=0.5, alpha=0.05)
            ax2.plot(t, td_imag, c='tab:red', lw=0.5, alpha=0.05)

    F_ABS_MEAN = numpy.mean(F_ABS[1:,], axis=0) # [1:,] so to remove zeros in begining
    F_REAL_MEAN = numpy.mean(F_REAL[1:,], axis=0)
    F_IMAG_MEAN = numpy.mean(F_IMAG[1:,], axis=0)
    T_ABS_MEAN = numpy.mean(T_ABS[1:,], axis=0)
    T_REAL_MEAN = numpy.mean(T_REAL[1:,], axis=0)
    T_IMAG_MEAN = numpy.mean(T_IMAG[1:,], axis=0)

    F_ABS_MIN = numpy.min(F_ABS[1:,], axis=0)
    F_ABS_MAX = numpy.max(F_ABS[1:,], axis=0)

    F_ABS = F_ABS[1:,]
    T_ABS = T_ABS[1:,]

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

        # ax1.plot(f, F_REAL_MEAN, c='tab:green', ls='--', alpha=0.75, lw=1.5, label="Mean real")
        # ax1.plot(f, F_IMAG_MEAN, c='tab:red',ls='dotted', alpha=0.75, lw=1.5, label="Mean imaginary")
        ax1.plot([f[0], f[-1]], [0, 0], c='k', lw=0.5, alpha=0.5, ls='dotted')
        XTICKS = numpy.asarray([-26, -13, 0, 13, 26]) * Fd/1000 # kHz
        XVALS = [-26, -13, 0, 13, 26]
        ax1.set_xticks(XVALS)
        ax1.set_xticklabels(XTICKS)
        ax1.set_xlim([-26.5, 26.5])
        # ax1.legend(ncol=3, loc='lower center')
        #ax1.grid(which='major')
        ax1.set_ylim([-0.25*numpy.amax(F_ABS_MEAN), 1.25*numpy.amax(F_ABS_MEAN)])


        ax2.set_title("Impulse Response")
        ax2.set_xlabel("$t \ [\mu s]$")
        ax2.set_ylabel("$h(\\tau)$")
        ax2.plot(t, T_ABS_MEAN, c='k', lw=2, alpha=1, label="Mean absolute")
        ax2.plot(t, T_REAL_MEAN, c='tab:green', ls='--', alpha=0.75, lw=1.5, label="Mean real")
        ax2.plot(t, T_IMAG_MEAN, c='tab:red', ls='dotted', alpha=0.75, lw=1.5, label="Mean imaginary")
        ax2.plot([t[0], t[-1]], [0, 0], c='k', lw=0.5, alpha=0.5, ls='dotted')
        #ax2.legend(ncol=3, loc='lower center')
        ax2.set_xlim([numpy.amin(t), numpy.amax(t)])
        #ax2.set_ylim([-1.5*numpy.amax(T_ABS_MEAN), 1.25*numpy.amax(T_ABS_MEAN)])

        fig.tight_layout(pad=2)

    return F_ABS_MEAN, F_ABS, T_ABS_MEAN, T_ABS, t



def PDPDelaySpreads(X, Y, range_slices, plot=False):
    F_ABS_MEAN, F_ABS, T_ABS_MEAN, T_ABS, t = TransferFunctionImpulseResponse(X, Y, range_slices)
    F_SAMPLES = len(T_ABS_MEAN)
    T_SAMPLES = len(T_ABS)

    PDPS = numpy.zeros(F_SAMPLES)
    START_DELAYS = []
    MAX_DELAYS = []
    MEAN_DELAYS = []
    RMS_DELAYS = []

    for i in range(T_SAMPLES):
        P = PowerDelayProfile(T_ABS[i])
        T_s, T_m = MaximumDelay(T_ABS[i])
        mean = MeanDelay(T_ABS[i])
        rms = RMSDelay(T_ABS[i])

        PDPS = numpy.vstack((PDPS, P))
        START_DELAYS.append(T_s)
        MAX_DELAYS.append(T_m)
        MEAN_DELAYS.append(mean)
        RMS_DELAYS.append(rms)

    # P = PowerDelayProfile(T_ABS_MEAN)
    # T_s, MAX_DELAY = MaximumDelay(T_ABS_MEAN)
    # MEAN_DELAY = MeanDelay(T_ABS_MEAN)
    # RMS_DELAY = RMSDelay(T_ABS_MEAN)
    PDPS = PDPS[1:,]
    P = numpy.mean(PDPS, axis=0)
    T_s = numpy.mean(START_DELAYS)
    MAX_DELAY = numpy.mean(MAX_DELAYS)
    MEAN_DELAY = numpy.mean(MEAN_DELAYS)
    RMS_DELAY = numpy.mean(RMS_DELAYS)

    T_MAX = numpy.amax(t)
    T_NUM = len(t)

    # Compute true delay times.
    true_start_delay = T_MAX * T_s/T_NUM # [microseconds]
    true_maximum_delay = T_MAX * (MAX_DELAY-T_s)/T_NUM # [microseconds]
    true_mean_delay = T_MAX * (MEAN_DELAY)/T_NUM # [microseconds]
    true_rms_delay = T_MAX * (RMS_DELAY)/T_NUM # [microseconds]

    true_start_delay_plot = T_MAX * T_s/T_NUM # [microseconds]
    true_maximum_delay_plot = T_MAX * (MAX_DELAY)/T_NUM # [microseconds]
    true_mean_delay_plot = T_MAX * (MEAN_DELAY+T_s)/T_NUM # [microseconds]
    true_rms_delay_plot = T_MAX * (RMS_DELAY+T_s)/T_NUM # [microseconds]

    # Plot in figure.
    if plot:
        # Prepare string to print in plot.
        true_start_delay_str = "$\\bar T_s = $" + str(numpy.around(true_start_delay, 5)) + " $[\mu s]$"
        true_maximum_delay_str = "$\\bar T_m = $" + str(numpy.around(true_maximum_delay, 5)) + " $[\mu s]$"
        true_mean_delay_str = "$\\bar \mu_{T_m} = $" + str(numpy.around(true_mean_delay, 5)) + " $[\mu s]$"
        true_rms_delay_str = "$\\bar \sigma_{T_m} = $" + str(numpy.around(true_rms_delay, 5)) + " $[\mu s]$"

        ymax = numpy.amax(P)
        ystr = -ymax * 0.4
        fig = plt.figure(figsize=(8.2, 4.2))
        matplotlib.rcParams.update({'font.size': 14})
        #plt.suptitle("Average of Power Delay Profiles")
        ax = fig.add_subplot()
        for i in range(T_SAMPLES):
            ax.plot(t, PDPS[i], c='k', lw=0.5, alpha=0.05)
        # Plot Power Delay Profile and 0-line.
        marksize = 50
        ax.set_title("Average of Power Delay Profiles")
        ax.plot(t, P,
            label="Average of $P(\\tau)$", c='k', linestyle='-', lw=1.5)
        ax.fill_between(t, numpy.amax(PDPS[1:,], axis=0), numpy.amin(PDPS[1:,], axis=0), alpha=0.15, color='k', label="Maximum/Minimum of $P(\\tau)$")
        ax.plot([0, numpy.amax(t)], [0, 0], c='k', lw=0.5, alpha=.5, ls='dotted')
        ax.scatter(true_start_delay_plot, 0,
            c='k', marker='x', s=marksize) #,label="Start time $T_s$")
        ax.text(true_start_delay, ystr, true_start_delay_str)
        # Plot maximum delay.
        ax.scatter(true_maximum_delay_plot, 0,
            c='k', marker='x', s=marksize) #,label="Maximum delay $T_m$")
        ax.text(true_maximum_delay_plot, 4*ystr, true_maximum_delay_str)
        # Plot mean delay.
        ax.scatter(true_mean_delay_plot, 0,
            c='k', marker='x', s=marksize) #,label="Mean delay $\mu_{T_m}$")
        ax.text(true_mean_delay_plot, 2*ystr, true_mean_delay_str)
        # Plot RMS delay.
        ax.scatter(true_rms_delay_plot, 0,
            c='k', marker='x', s=marksize)#, label="RMS delay spread $\sigma_{T_m}$")
        ax.text(true_rms_delay_plot, 3*ystr, true_rms_delay_str)
        ax.set_xlabel("$\\tau \ [\mu s]$")
        ax.set_ylabel("$\\bar P(\\tau)$")
        ax.set_xlim([0, 2*(true_start_delay+true_maximum_delay)])
        ylim = [-1.75*ymax, 2*ymax]
        ax.set_ylim(ylim)
        ax.plot([true_start_delay, true_start_delay], ylim, lw=0.5, c='k', alpha=0.5, ls='dotted')
        ax.legend(ncol=2, loc='upper center')
        fig.tight_layout(pad=2)

    return true_start_delay, true_maximum_delay, true_mean_delay, true_rms_delay



def LengthScaleGP(X, Y, range_slices, plot=False):
    T_SAMPLES = 52
    LS = []

    # Version 1, GP for all 10 slices.
    for slice in range_slices:
        x = numpy.array([])
        y = numpy.array([])
        t_slice_np = data.get_slice(xy='x', vector=X, time=slice, freq=None)
        f = t_slice_np[:,0]
        Y_slice_np = data.get_slice(xy='y', vector=Y, time=slice, freq=None)
        fd = (Y_slice_np[:,0] + (1j)*Y_slice_np[:,1])
        abs = numpy.absolute(fd)
        x = numpy.append(x, f)
        y = numpy.append(y, abs)

    # # Version 2, use only 1 of the 10 slices, the middle slice
    # slice = range_slices[5]
    # t_slice_np = data.get_slice(xy='x', vector=X, time=slice, freq=None)
    # f = t_slice_np[:,0]
    # Y_slice_np = data.get_slice(xy='y', vector=Y, time=slice, freq=None)
    # fd = (Y_slice_np[:,0] + (1j)*Y_slice_np[:,1])
    # abs = numpy.absolute(fd)
    # x = f
    # y = abs

    # Version 3, GP for each slice (10), then take mean

        f_train = torch.from_numpy(x).reshape(-1,1).type('torch.FloatTensor').contiguous()
        abs_train = torch.from_numpy(y).reshape(-1,).type('torch.FloatTensor').contiguous()

        EPOCHS = 1500 # number of iterations during training
        LR = 0.025 # learning rate during training

        gpytorch.settings.skip_posterior_variances(state=False)
        gpytorch.settings.lazily_evaluate_kernels(state=False)
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = GP(f_train, abs_train, likelihood)
        loss, ls, ns = model.optimize(f_train, abs_train, likelihood, EPOCHS, LR)

        print("Loss:", loss.item())
        print("Lengthscale:", ls)

        # PRED_STEPS = 100
        # PRED = 26
        # xpred = torch.linspace(-PRED, PRED, PRED_STEPS)
        # mean, var, lower, upper, cov = model.predict(xpred, likelihood)
        #
        # fig = plt.figure(figsize=(8, 4))
        # matplotlib.rcParams.update({'font.size': 14})
        # plt.suptitle("GPR of aboslute value (" + str(len(range_slices)) + " sampled transfer functions")
        # ax = fig.add_subplot()
        # # Plot Power Delay Profile and 0-line.
        # ax.scatter(f_train, abs_train,
        #     label="Training data", c='k', marker='.', alpha=0.1)
        # ax.plot(xpred, mean,
        #     label="Mean", c='k', lw=2)
        # ax.fill_between(xpred, lower.numpy(), upper.numpy(),
        #     alpha=0.2, color='k', label="Variance (95%)")
        #
        # plt.show()


        LS.append(ls) # version 3

        # Plot in figure.
        if plot:
            PRED_STEPS = 200
            PRED = 26
            xpred = torch.linspace(-PRED, PRED, PRED_STEPS)
            mean, var, lower, upper, cov = model.predict(xpred, likelihood)

            fig = plt.figure(figsize=(8, 4))
            matplotlib.rcParams.update({'font.size': 14})
            plt.suptitle("GPR of aboslute value (" + str(len(range_slices)) + " sampled transfer functions")
            ax = fig.add_subplot()
            # Plot Power Delay Profile and 0-line.
            ax.scatter(f_train, abs_train,
                label="Training data", c='k', marker='.', alpha=0.1)
            ax.plot(xpred, mean,
                label="Mean", c='k', lw=2)
            ax.fill_between(xpred, lower.numpy(), upper.numpy(),
                alpha=0.2, color='k', label="Variance (95%)")
            ax.text(0, 0+0.05, "Length-scale $\ell = $" + str(numpy.around(312.5*ls,5)), ha='center')

            XTICKS = numpy.asarray([-26, -13, 1, 13, 26]) * 312.5 # kHz
            XVALS = [-26, -13, 1, 13, 26]
            ax.set_xticks(XVALS)
            ax.set_xticklabels(XTICKS)
            ax.set_xlabel("$f\ [kHz]$")
            ax.set_ylabel("$|H(f)|$")
            ax.set_xlim([-PRED, PRED])
            ymax = numpy.amax(mean.numpy())
            ax.set_ylim([0, 1.5*ymax])
            ax.legend(ncol=3)
            fig.tight_layout(pad=2)
            ax.legend(ncol=3)

            # Covariance matrix.
            fig_cov = plt.figure(figsize=(5, 5))
            plt.suptitle("Covariance matrix")
            ax_cov = fig_cov.add_subplot()
            cmap = 'jet'
            ax_cov.pcolormesh(cov.detach().numpy(), cmap=cmap)
            ax_cov.set_xticklabels([])
            ax_cov.set_yticklabels([])
            plt.gca().invert_yaxis()

    ls = numpy.mean(LS)
    return ls # Version 1 och 2 return ls

def LengthScaleGPGrid(X, Y, range_slices_grid, plot=False):
    SAMPLES = len(range_slices_grid)
    ITERS = len(range_slices_grid[0])
    S = range(1, SAMPLES+1)

    START_D = []
    MAX_D = []
    MEAN_D = []
    RMS_D = []
    LS = []

    freq = 312.5 # [kHz]

    for i in range(SAMPLES):
        curr_slice = range_slices_grid[i]
        print(curr_slice)
        start_delay, max_delay, mean_delay, rms_delay = PDPDelaySpreads(X, Y, curr_slice, plot=False)
        ls = LengthScaleGP(X, Y, curr_slice, plot=False)

        START_D.append(start_delay)
        MAX_D.append(max_delay)
        MEAN_D.append(mean_delay)
        RMS_D.append(rms_delay)
        LS.append(ls)

    LS_MEAN = numpy.mean(LS)
    MAX_MEAN = numpy.mean(MAX_D)
    MEAN_MEAN = numpy.mean(MEAN_D)
    RMS_MEAN = numpy.mean(RMS_D)

    LS = numpy.multiply(LS, freq)
    LS_STD = numpy.std(LS)

    str_ls = "Mean $\\ell=$" + str(numpy.around(freq*LS_MEAN, 5))
    str_max = "Mean $T_m=$" + str(numpy.around(MAX_MEAN, 5)) + " $[\\mu s]$"
    str_mean = "Mean $\\mu_{T_m}=$" + str(numpy.around(MEAN_MEAN, 5)) + " $[\\mu s]$"
    str_rms = "Mean $\\sigma_{T_m}=$" + str(numpy.around(RMS_MEAN, 5)) + " $[\\mu s]$"

    MAX_D_T = MAX_D
    MEAN_D_T = MEAN_D
    RMS_D_T = RMS_D

    START_D = numpy.divide(START_D, 1000) # [milliseconds]
    MAX_D = numpy.divide(MAX_D, 1000)
    MEAN_D = numpy.divide(MEAN_D, 1000)
    RMS_D = numpy.divide(RMS_D, 1000)

    MAX_D = numpy.divide(1, MAX_D) # approx freq [Hz]
    MEAN_D = numpy.divide(1, MEAN_D)
    RMS_D = numpy.divide(1, RMS_D)

    C_MAX = numpy.divide(MAX_D, LS) # ratio
    C_MEAN = numpy.divide(MEAN_D, LS)
    C_RMS =  numpy.divide(RMS_D, LS)

    C_MAX_MEAN = numpy.mean(C_MAX) # mean ratio
    C_MEAN_MEAN = numpy.mean(C_MEAN)
    C_RMS_MEAN = numpy.mean(C_RMS)

    C_MAX_STD = numpy.std(C_MAX)
    C_RMS_STD = numpy.std(C_RMS)

    str_cmax = "Mean $k_{T_m}$: " + str(numpy.around(C_MAX_MEAN, 5))
    str_cmean = "Mean ratio: " + str(numpy.around(C_MEAN_MEAN, 5))
    str_crms = "Mean $k_{\\sigma_{T_m}}$: " + str(numpy.around(C_RMS_MEAN, 5))

    str_cmaxstd = "Std $k_{T_m}$: " + str(numpy.around(C_MAX_STD, 5))
    str_crmsstd = "Std $k_{\\sigma_{T_m}}$: " + str(numpy.around(C_RMS_STD, 5))

    matplotlib.rcParams.update({'font.size': 14})
    fig = plt.figure(figsize=(8, 5))
    fig.suptitle(DATA_SET)
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.set_title("$k_{T_m}$ for each sample set")
    ax1.plot(S, C_MAX,
        c='k', lw=2, marker='o')
    for i in range(len(C_MAX)):
        ax1.text(i+1, C_MAX[i]+0.5, str(numpy.around(C_MAX[i], 2)), ha='center')

    ax1.set_ylim([0, 1.35*numpy.amax(C_MAX)])
    ax1.set_xticks(S)
    ax1.set_xticklabels(S)
    ax1.set_xlabel("Sample set")
    ax1.set_ylabel("$k_{T_m}$")
    ax1.text(1, 0.1*numpy.amax(C_MAX), str_ls + "\n" + str_max)
    ax1.text(S[-1], 0.1*numpy.amax(C_MAX), str_cmax + "\n" + str_cmaxstd, ha="right")

    # ax2 = fig.add_subplot(3, 1, 2)
    # ax2.set_title("Ratio $\\ell/\\mu_{\\tau}$ vs. data set")
    # ax2.plot(S, C_MEAN,
    #     c='k', lw=2, marker='o')
    # ax2.set_ylim([0, 1.25*numpy.amax(C_MEAN)])
    # ax2.set_xticks(S)
    # ax2.set_xticklabels(S)
    # ax2.set_xlabel("Data set")
    # ax2.set_ylabel("Ratio $\\ell/\\mu_{\\tau}$")
    # ax2.text(1, 0.1*numpy.amax(C_MEAN), str_ls + "\n" + str_mean)
    # ax2.text(S[-1], 0.1*numpy.amax(C_MEAN), str_cmean, ha="right")

    ax3 = fig.add_subplot(2, 1, 2)
    ax3.set_title("$k_{\\sigma_{T_m}}$ for each sample set")
    ax3.plot(S, C_RMS,
        c='k', lw=2, marker='o')
    for i in range(len(C_RMS)):
        ax3.text(i+1, C_RMS[i]+1, str(numpy.around(C_RMS[i], 2)), ha='center')
    ax3.set_ylim([0, 1.35*numpy.amax(C_RMS)])
    ax3.set_xticks(S)
    ax3.set_xticklabels(S)
    ax3.set_xlabel("Sample set")
    ax3.set_ylabel("$k_{\\sigma_{T_m}}$")
    ax3.text(1, 0.1*numpy.amax(C_RMS), str_ls + "\n" + str_rms)
    ax3.text(S[-1], 0.1*numpy.amax(C_RMS), str_crms + "\n" + str_crmsstd, ha="right")

    fig.tight_layout(pad=2)

    return LS_MEAN, MAX_MEAN, MEAN_MEAN, RMS_MEAN, LS, MAX_D_T, MEAN_D_T, RMS_D_T


def AutoCorr(X, Y, a, range_slices_grid, plot=False):
    M = 53
    N = 10
    Bc_all = []
    for subset in range_slices_grid:                            # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        s = numpy.zeros(M, dtype=numpy.complex)
        for k in range(0, M):
            for n in range(0, N):
                t_slice_np = data.get_slice(xy='x', vector=X, time=subset[n], freq=None)
                f = t_slice_np[:,0]
                Y_slice_np = data.get_slice(xy='y', vector=Y, time=subset[n], freq=None)
                fd = (Y_slice_np[:,0] + (1j)*Y_slice_np[:,1])
                # Insert zero.
                ls = fd[25]
                rs = fd[27]
                mean = (ls+rs)/2
                f = numpy.insert(f, 26, 0)
                fd = numpy.insert(fd, 26, mean)
                for m in range(0, M-k):
                    s[k] += (numpy.conjugate(fd[m]) * fd[m+k])
            s[k] /= 1/(N*(M-k))

        s_abs = numpy.absolute(s)
        s_norm = s_abs/s_abs[0]

        ###########################

        k = M-1
        Bc = 0
        K = 0
        while k >= 0:
            if s_norm[k] > a:
                Bc = k * 312.5
                K = k
                break
            k -= 1

        Bc_all.append(Bc)

        XTICKS = numpy.asarray([-26, -13, 0, 13, 26])
        XTICKLABELS = (26+XTICKS) * 312.5
        f = data.get_freq()
        f = numpy.insert(f, 26, 0)
        result = s_norm
        found_ind = K

        if plot:
            matplotlib.rcParams.update({'font.size': 14})
            fig = plt.figure(figsize=(8, 5))
            fig.suptitle(DATA_SET)
            ax1 = fig.add_subplot(1, 1, 1)
            ax1.set_title("Autocorrelation of transfer function")
            ax1.plot(f, result,
                c='k', lw=1.5)
            #ax1.scatter(f[ind], result[ind], c='k', label="Max")
            ax1.scatter(f[found_ind], result[found_ind], c='k', label="Max")
            ax1.text(f[found_ind], result[found_ind], Bc)
            ax1.set_xlabel("$\Delta f$")
            ax1.set_ylabel("Autocorrelation")
            ax1.set_xticks(XTICKS)
            ax1.set_xticklabels(XTICKLABELS)

    return Bc_all

    # return Bc_all

    # CB_ALL = []
    # for range_slices in range_slices_grid:
    #     RES = numpy.zeros(52)
    #     for slice in range_slices:
    #         t_slice_np = data.get_slice(xy='x', vector=X, time=slice, freq=None)
    #         f = t_slice_np[:,0]
    #         Y_slice_np = data.get_slice(xy='y', vector=Y, time=slice, freq=None)
    #         fd = (Y_slice_np[:,0] + (1j)*Y_slice_np[:,1])
    #         #print(fd)
    #         result = numpy.correlate(fd, fd, mode='full')
    #         result = result[result.size//2:]
    #         RES = numpy.vstack((RES, result))
    #
    #     RES = RES[1:,]
    #     RES_MEAN = numpy.mean(RES, axis=0)
    #     result = numpy.absolute(RES_MEAN)
    #     val = numpy.amax(result)
    #     ind = numpy.argmax(result)
    #     find = a*val
    #     found_ind = ind
    #     found_val = val
    #     for i in range(ind, result.size):
    #         if found_val <= find:
    #             found_ind = i
    #             break
    #         else:
    #             found_val = result[i]
    #
    #     print(found_ind)
    #     CB = (found_ind-ind)*312.5
    #     CB_ALL.append(CB)

    #     XTICKS = numpy.asarray([-26, -13, 0, 13, 26])
    #     XTICKLABELS = (26+XTICKS) * 312.5
    #
    #     if plot:
    #         matplotlib.rcParams.update({'font.size': 14})
    #         fig = plt.figure(figsize=(8, 5))
    #         fig.suptitle(DATA_SET)
    #         ax1 = fig.add_subplot(1, 1, 1)
    #         ax1.set_title("Autocorrelation of transfer function")
    #         ax1.plot(f, result,
    #             c='k', lw=1.5)
    #         ax1.scatter(f[ind], result[ind], c='k', label="Max")
    #         ax1.scatter(f[found_ind], result[found_ind], c='k', label="Max")
    #         ax1.text(f[found_ind], result[found_ind], CB)
    #         ax1.set_xlabel("$\Delta f$")
    #         ax1.set_ylabel("Autocorrelation")
    #         ax1.set_xticks(XTICKS)
    #         ax1.set_xticklabels(XTICKLABELS)
    #
    # CB_MEAN = numpy.mean(CB_ALL)
    # CB_STD = numpy.std(CB_ALL)
    # return CB_MEAN, CB_STD

def AutoCorr2(X, Y, a, range_slices_grid, plot=False):
    CB_ALL = []
    for range_slices in range_slices_grid:
        RES = numpy.zeros(52)
        for slice in range_slices:
            t_slice_np = data.get_slice(xy='x', vector=X, time=slice, freq=None)
            f = t_slice_np[:,0]
            Y_slice_np = data.get_slice(xy='y', vector=Y, time=slice, freq=None)
            fd = (Y_slice_np[:,0] + (1j)*Y_slice_np[:,1])
            #print(fd)
            result = numpy.correlate(fd, fd, mode='full')
            result = result[result.size//2:]
            RES = numpy.vstack((RES, result))

        RES = RES[1:,]
        RES_MEAN = numpy.mean(RES, axis=0)
        result = numpy.absolute(RES_MEAN)
        val = numpy.amax(result)
        ind = numpy.argmax(result)
        find = a*val
        found_ind = ind
        found_val = val
        for i in range(ind, result.size):
            if found_val <= find:
                found_ind = i
                break
            else:
                found_val = result[i]

        print(found_ind)
        CB = (found_ind-ind)*312.5
        CB_ALL.append(CB)

        XTICKS = numpy.asarray([-26, -13, 0, 13, 26])
        XTICKLABELS = (26+XTICKS) * 312.5

        if plot:
            matplotlib.rcParams.update({'font.size': 14})
            fig = plt.figure(figsize=(8, 5))
            fig.suptitle(DATA_SET)
            ax1 = fig.add_subplot(1, 1, 1)
            ax1.set_title("Autocorrelation (NUMPY) of transfer function")
            ax1.plot(f, result,
                c='k', lw=1.5)
            ax1.scatter(f[ind], result[ind], c='k', label="Max")
            ax1.scatter(f[found_ind], result[found_ind], c='k', label="Max")
            ax1.text(f[found_ind], result[found_ind], CB)
            ax1.set_xlabel("$\Delta f$")
            ax1.set_ylabel("Autocorrelation")
            ax1.set_xticks(XTICKS)
            ax1.set_xticklabels(XTICKLABELS)

    CB_MEAN = numpy.mean(CB_ALL)
    CB_STD = numpy.std(CB_ALL)
    return CB_MEAN, CB_STD

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
SKIP = 1000 # number of time samples to skip over in the beginning
SKIP_CLOSE = 0 # if time sample is within SKIP_CLOSE to previous time sample imported, skip it
FILE = FILEs[4] # file to import from
MAC_ID = MAC_IDs[1]
# V1: SKIP=500
# D5: SKUP=1000    41

# 190331 190413 190329 190412

# D1: 03 - JUMP=0 SKIP=0 SKIP_CLOSE=0
# D2: 12 - JUMP=0 SKIP=500 SKIP_CLOSE=0
# D3: 13 - JUMP0 SKIP=0 SKIP_CLOSE=0
# D4: 20 - JUMP0 SKIP=0 SKIP_CLOSE=0
# D5: 41 - JUMP0 SKIP=1000 SKIP_CLOSE=0

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
RANGE_SLICES = range(1,51)
SAMPLE_SET = 10
DATA_SET = "Data set 1"

# Plot one GPR.
#LengthScaleGP(X,Y, RANGE_SLICES, plot=True)


TransferFunctionImpulseResponse(X, Y, RANGE_SLICES, plot=True)
PDPDelaySpreads(X, Y, RANGE_SLICES, plot=True)

RANGE_SLICES_GRID = numpy.asarray(RANGE_SLICES)
RANGE_SLICES_GRID = RANGE_SLICES_GRID.reshape(-1,SAMPLE_SET)
# LS_MEAN, MAX_MEAN, MEAN_MEAN, RMS_MEAN, LS, MAX_D, MEAN_D, RMS_D = LengthScaleGPGrid(X, Y, range_slices_grid=RANGE_SLICES_GRID)
# print("LS:", LS)
# print("MAX_D:", MAX_D)
# print("RMS_D:", RMS_D)

a = 0.75
Bc_all = AutoCorr(X, Y, a, RANGE_SLICES_GRID, plot=True)
# CB_MEANS, CB_STDS = AutoCorr2(X, Y, a, RANGE_SLICES_GRID, plot=True)
print(Bc_all)
print(numpy.mean(Bc_all))
# print("CB_MEANS", CB_MEANS)
# print("CB_STDS", CB_STDS)

RES_MAX = [3.35664, 3.78825, 3.80758, 3.40936, 3.21272]
RES_RMS = [12.44096, 11.40379, 15.72213, 13.84128, 10.30898]

RES_MAX2 = [3.08446, 3.87027, 3.44465, 3.36641, 2.95381]
RES_RMS2 = [11.41688, 11.66046, 14.20538, 13.67092, 9.51897]

RES_MAX3 = [4.05, 4.22, 3.56, 3.66, 3.81,
            3.21, 3.04, 3.34, 3.09, 3.34,
            #3.55, 3.39, 3.68, 3.15, 3.59,
            3.20, 3.07, 3.68, 2.97, 3.38,
            3.33, 3.57, 3.36, 3.38, 3.83,
            3.23, 3.16, 3.15, 3.42, 3.32]
RES_RMS3 = [12.08, 12.36, 11.09, 11.7, 12.21,
            12.41, 11.81, 12.4, 11.22, 12.52,
            #15.9, 14.41, 13.83, 14.61, 16.19,
            12.76, 12.88, 13.81, 12.57, 14.6,
            13.19, 13.91, 13.63, 13.1, 14.94,
            11.37, 10.98, 11.12, 12.02, 11.54]

print("Mean k_max: ", numpy.mean(RES_MAX3))
print("Std k_max:  ", numpy.std(RES_MAX3))

print("Mean k_rms: ", numpy.mean(RES_RMS3))
print("Std k_rms:  ", numpy.std(RES_RMS3))


""" RESULTS

    k_max =   3.51 +- 0.24
    k_rms =  12.74 +- 1.89

    k_max2 =  3.34 +- 0.32
    k_rms2 = 12.09 +- 1.69

    k_max3 = 3.46  +- 0.29
    k_rms3 = 12.82 +- 1.47

    k_max3 = 3.41  +- 0.31
    k_rms3 = 12.49 +- 1.06


"""

""" RESULTS
03
LS: [ 995.14307827  970.59096396 1010.29168814  987.77162284  997.39519507]
MAX_D: [0.27562499999999995, 0.260859375, 0.280546875, 0.29039062499999996, 0.26578125]
RMS_D: [0.09046695505041047, 0.08986897453772555, 0.09249239416926215, 0.09234028211585613, 0.08903810356424947]

12
LS: [2133.50018859 2183.12309682 2168.86515915 2142.10064709 2111.34392023]
MAX_D: [0.1575, 0.15750000000000006, 0.152578125, 0.1575, 0.15257812499999998]
RMS_D: [0.040716241802448275, 0.04053558489221651, 0.04109500976648475, 0.04331003667531423, 0.04077285268066998]

13
LS: [1302.1967113  1310.59792638 1302.2371009  1313.53552639 1314.49012458]
MAX_D: [0.25593750000000004, 0.2559375, 0.260859375, 0.2559375, 0.260859375]
RMS_D: [0.06580743601649784, 0.06201623611974481, 0.06802449934092618, 0.05923749703560124, 0.06365809206667823]

20
LS: [1872.59072065 1765.05059004 1790.29785097 1866.33890867 1809.96096134]
MAX_D: [0.16734375, 0.172265625, 0.16734374999999996, 0.16734375, 0.17718749999999997]
RMS_D: [0.0428606878645995, 0.04265272000148329, 0.04012413025625115, 0.04080035374207711, 0.042799984942214535]

41
LS: [1169.81332749 1144.96777952 1199.9072507  1164.35173154 1181.84815347]
MAX_D: [0.2953125, 0.2953125, 0.2953125, 0.2953124999999999, 0.300234375]
RMS_D: [0.08377730602486874, 0.08503555032828115, 0.08362687718451003, 0.08403484449590137, 0.08643871819968986]

"""







plt.show()
