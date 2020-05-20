""" gpr1d1_coherence_bandwidth.py

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
    1)   Computing time spread by Discrete Inverse Fourier Transform.
    2)   Looking in posterior covariance matrix of a GP.
    3)   Computing auto-correlation of estimated mean.

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





""" LOAD DATA
    Initalize data. Load data. Prepare data and put into vectors.

    Version 1: Version 1: LIMIT=102 JUMP=0 SKIP=500 SKIP_CLOSE=0.25 MAC_ID="8c:85:90:a0:2a:6f"
"""
FILEs = ["data/wifi_capture_190331.csv" , "data/wifi_capture_190413.csv"]
MAC_IDs = ["8c:85:90:a0:2a:6f", "1c:b7:2c:d7:67:ec", "ff:ff:ff:ff:ff:ff",
           "34:a3:95:c3:57:10", "a0:93:51:72:24:0f"]
LIMIT = 52 # number of time samples to import
JUMP = 0 # number of time samples to jump over after each imported time sample
SKIP = 500 # number of time samples to skip over in the beginning
SKIP_CLOSE = 0 # if time sample is within SKIP_CLOSE to previous time sample imported, skip it
FILE = FILEs[0] # file to import from
MAC_ID = MAC_IDs[0]

data = Data(name="Coherence bandwidth") # initalie data object
data.load(csv_file=FILE, mac_str=MAC_ID, limit=LIMIT, skip=SKIP,
          jump=JUMP, skip_close=SKIP_CLOSE) # load data
data.prep() # prepare data and put into vectors
#data.norm_effect() # normalize so energy is consistent
# plot3D(data, "Before phase compensation")
# data.norm_phase()
# plot3D(data, "After phase compensation")

X_np = data.get_x() # get imported input points
Y_np = data.get_y() # get imported output points




""" TEST IFFT AND FFT
    Test to see how numpy.ifft and numpy.fft behaves.
"""
def test_ifft():
    T_START = 0
    T_END = 10
    T_SAMPLES = 100
    Ts = (T_END-T_START)/T_SAMPLES
    Fs = 1/Ts
    t = numpy.linspace(T_START, T_END, T_SAMPLES)
    a1 = 1
    f1 = 1/2
    a2 = 0.5
    f2 = 1/8
    td = a1*numpy.sin(2*numpy.pi*f1*t) + a2*numpy.sin(2*numpy.pi*f2*t)
    fd = numpy.fft.fft(td)

    fig = plt.figure(figsize=(8, 10))
    matplotlib.rcParams.update({'font.size': 12})

    ax1 = fig.add_subplot(5, 1, 1)
    ax1.set_title("Time Domain\n" + str(a1) + "$\cdot\sin(2\pi\cdot$" +str(f1) + "t)" + " + " + str(a2) + "$\cdot\sin(2\pi\cdot$" + str(f2) + "t)" +
        "\n $T_s=$" + str(Ts) + ", samples: " + str(T_SAMPLES))
    ax1.plot(t, td, c='k')
    ax1.plot(t, td, c='k')
    ax1.set_xlabel("$t [s]$")

    f = t - numpy.amin(t)
    f = f / numpy.amax(f)
    f = f - 0.5
    ax2 = fig.add_subplot(5, 1, 2)
    ax2.set_title("Frequency Domain with numpy.fft.ifft() directly")
    ax2.plot(f, numpy.absolute(fd), c='b', lw=1, label="Aboslute value")
    ax2.plot(f, numpy.real(fd), c='g', lw=1, label="Real part", alpha=0.5)
    ax2.plot(f, numpy.imag(fd), c='r', lw=1, label="Imaginary part", alpha=0.5)
    ax2.set_ylabel("$\\frac{1}{T_s}=$" + str(1/Ts))
    XVALS2 = numpy.linspace(-0.5, 0.5, 5)
    XTICKS2 = XVALS2 * Fs
    ax2.set_xticks(XVALS2)
    ax2.set_xticklabels(XTICKS2)
    ax2.set_xlabel("$f [Hz]$")
    ax2.legend()

    fd = numpy.roll(fd, shift=T_SAMPLES//2)
    ax3 = fig.add_subplot(5, 1, 3)
    ax3.set_title("Frequency Domain with numpy.fft.ifft() and numpy.roll(Ts/2)")
    ax3.plot(f, numpy.absolute(fd), c='b', lw=1, label="Absolute value")
    ax3.plot(f, numpy.real(fd), c='g', lw=1, label="Real part", alpha=0.5)
    ax3.plot(f, numpy.imag(fd), c='r', lw=1, label="Imaginary part", alpha=0.5)
    ax3.set_ylabel("$\\frac{1}{T_s}=$" + str(1/Ts))
    XVALS3 = numpy.linspace(-0.5, 0.5, 5)
    XTICKS3 = XVALS3 * Fs
    ax3.set_xticks(XVALS3)
    ax3.set_xticklabels(XTICKS3)
    ax3.set_xlabel("$f [Hz]$")
    ax3.legend()

    t2 = (f - numpy.amin(f))
    t2 = t2/Ts
    td2 = numpy.fft.ifft(fd)
    ax4 = fig.add_subplot(5, 1, 4)
    ax4.set_title("Time Domain, IFFT of above")
    ax4.plot(t2, td2, c='k')
    ax4.plot(t2, td2, c='k')
    ax4.set_xlabel("$t [s]$")

    td2 = numpy.fft.ifft(numpy.roll(fd, shift=T_SAMPLES//2))
    ax5 = fig.add_subplot(5, 1, 5)
    ax5.set_title("Time Domain, IFFT of above with first using roll()")
    ax5.plot(t2, td2, c='k')
    ax5.plot(t2, td2, c='k')
    ax5.set_xlabel("$t [s]$")

    fig.tight_layout(pad=1)



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

def MaximumDelay(h, a=0.1):
    """ Finds the maximum delay T_m where P(tau = T_m) is lower than a set factor dP.
        dP is can be when P is lower than a*P, e.g. 0.1*P.

        However, since P is discrete and has a zero just after one time stamp, the
        total energy is calculated and then when a*tot_energy is reached, that is
        the maximum delay. It's interpreted in the "normal" sense, where T_m is the last
        longest path of significant energy.
    """
    P = PowerDelayProfile(h, norm=True) # get power delay profile of h, compensate for noise with norm=True
    P = P[0:40] # ignore last taus, defects?
    sum = numpy.sum(P) # get total energy in h
    dP = a * sum # the boundary to find, i.e. when the energi is below a * total energy in h
    for tau in range(len(P)): # iterate over each element in power delay profile
        if sum <= dP: # if the accumulated sum is below or equal to the boundary wanted...
            T_m = tau # ...save tau as T_m...
            break # ...and break.
        else:
            sum -= P[tau] # otherwise keep looking for the boundary
    return T_m

def MeanDelay(h):
    """ Find mean delay.
        mu_tau = frac{ sum_{tau_i} tau_i * P(tau_i) }
                     { sum_{tau_i} P(tau)           }

        Since noise is present, only the interval 0 to T_m is considered, i.e.
        we consider only the delays tau_i that have significant energy. The other
        tau_i above T_m are disregarded and considered non-existent.
    """
    P = PowerDelayProfile(h, norm=True)
    T_m = MaximumDelay(h)
    P = P[0:T_m+1] # consider only 0 to T_m, including T_m
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
    T_m = MaximumDelay(h)
    P = P[0:T_m+1]
    sum = 0
    for tau in range(len(P)):
        sum += numpy.power((tau - mu_tau), 2) * P[tau]
    div = sum/numpy.sum(P)
    sigma_tau = numpy.sqrt(div)
    return sigma_tau



""" TRANSFER FUNCTION AND IMPULSE RESPONSE AVERAGE OVER SLICES
"""
def ImpulseResponseAverage(range_slices, plot=False):
    if plot:
        matplotlib.rcParams.update({'font.size': 14})
        fig = plt.figure(figsize=(8.1, 6))
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2)

    T_SAMPLES = 52
    Fs = T_SAMPLES * 312.5 * 1000 # Hz
    Fd = 312.5 * 1000 # Hz
    Ts = 1/Fs

    F_ABS = numpy.zeros(T_SAMPLES)
    F_REAL = numpy.zeros(T_SAMPLES)
    F_IMAG = numpy.zeros(T_SAMPLES)
    T_ABS = numpy.zeros(T_SAMPLES)
    T_REAL = numpy.zeros(T_SAMPLES)
    T_IMAG = numpy.zeros(T_SAMPLES)

    for slice in range_slices:
        t_slice_np = data.get_slice(xy='x', vector=X_np, time=slice, freq=None)
        f = t_slice_np[:,0]
        Y_slice_np = data.get_slice(xy='y', vector=Y_np, time=slice, freq=None)
        fd = (Y_slice_np[:,0] + (1j)*Y_slice_np[:,1])

        td = numpy.fft.ifft(numpy.roll(fd, shift=T_SAMPLES//2))
        t = (f - numpy.amin(f))
        t = t * Ts * 1000000 # microseconds

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
            ax1.plot(f, fd_abs, c='tab:blue', lw=0.5, alpha=0.075)
            ax1.plot(f, fd_real, c='tab:green', lw=0.5, alpha=0.075)
            ax1.plot(f, fd_imag, c='tab:red', lw=0.5, alpha=0.075)

            ax2.plot(t, td_abs, c='tab:blue', lw=0.5, alpha=0.075)
            ax2.plot(t, td_real, c='tab:green', lw=0.5, alpha=0.075)
            ax2.plot(t, td_imag, c='tab:red', lw=0.5, alpha=0.075)

    F_ABS_MEAN = numpy.mean(F_ABS[1:,], axis=0) # [1:,] so to remove zeros in begining
    F_REAL_MEAN = numpy.mean(F_REAL[1:,], axis=0)
    F_IMAG_MEAN = numpy.mean(F_IMAG[1:,], axis=0)
    T_ABS_MEAN = numpy.mean(T_ABS[1:,], axis=0)
    T_REAL_MEAN = numpy.mean(T_REAL[1:,], axis=0)
    T_IMAG_MEAN = numpy.mean(T_IMAG[1:,], axis=0)

    if plot:
        matplotlib.rcParams.update({'font.size': 14})
        ax1.set_title("Transfer Function")
        ax1.set_xlabel("$f\ [kHz]$")
        ax1.set_ylabel("$|H(f)|$")
        ax2.set_title("Impulse Response")
        ax2.set_xlabel("$\\tau \ [\mu s]$")
        ax2.set_ylabel("$h(\\tau)$")
        ax1.plot(f, F_ABS_MEAN, c='tab:blue', lw=2, alpha=0.75, label="Mean absolute")
        ax1.plot(f, F_REAL_MEAN, c='tab:green', ls='--', alpha=0.75, lw=2, label="Mean real")
        ax1.plot(f, F_IMAG_MEAN, c='tab:red',ls='dotted', alpha=0.75, lw=2, label="Mean imaginary")
        ax2.plot(t, T_ABS_MEAN, c='tab:blue', lw=2, alpha=0.75, label="Mean absolute")
        ax2.plot(t, T_REAL_MEAN, c='tab:green', ls='--', alpha=0.75, lw=2, label="Mean real")
        ax2.plot(t, T_IMAG_MEAN, c='tab:red', ls='dotted', alpha=0.75, lw=2, label="Mean imaginary")
        XTICKS = numpy.asarray([-26, -13, 1, 13, 26]) * Fd/1000 # kHz
        XVALS = [-26, -13, 1, 13, 26]
        ax1.set_xticks(XVALS)
        ax1.set_xticklabels(XTICKS)
        ax1.legend(ncol=3)
        ax2.legend(ncol=3)
        ax1.set_xlim([numpy.amin(f), numpy.amax(f)])
        ax2.set_xlim([numpy.amin(t), numpy.amax(t)])
        fig.tight_layout(pad=1)

    return F_ABS_MEAN, F_REAL_MEAN, F_IMAG_MEAN, T_ABS_MEAN, T_REAL_MEAN, T_IMAG_MEAN, t



def PowerDelayProfileAverage(range_slices):
    F_ABS_MEAN, F_REAL_MEAN, F_IMAG_MEAN, T_ABS_MEAN, T_REAL_MEAN, T_IMAG_MEAN, t = ImpulseResponseAverage(range_slices)
    # T_ABS_MEAN[1:-1] = T_ABS_MEAN[0:-2]
    # T_ABS_MEAN[0] = 0
    P = PowerDelayProfile(T_ABS_MEAN)
    MAX_DELAY = MaximumDelay(T_ABS_MEAN)
    MEAN_DELAY = MeanDelay(T_ABS_MEAN)
    RMS_DELAY = RMSDelay(T_ABS_MEAN)
    T_MAX = numpy.amax(t)
    T_NUM = len(t)

    # Compute true delay times.
    true_maximum_delay = T_MAX * MAX_DELAY/T_NUM # [microseconds]
    true_mean_delay = T_MAX * MEAN_DELAY/T_NUM # [microseconds]
    true_rms_delay = T_MAX * RMS_DELAY/T_NUM # [microseconds]

    # Prepare string to print in plot.
    true_maximum_delay_str = "$T_m = $" + str(numpy.around(true_maximum_delay, 5)) + " $[\mu s]$"
    true_mean_delay_str = "$\mu_{\\tau} = $" + str(numpy.around(true_mean_delay, 5)) + " $[\mu s]$"
    true_rms_delay_str = "$\sigma_{\\tau} = $" + str(numpy.around(true_rms_delay, 5)) + " $[\mu s]$"

    Wc = 1000/(2*true_maximum_delay)
    Tm = true_maximum_delay

    # Plot in figure.
    ymax = numpy.amax(P)
    ystr = -ymax * 0.15
    fig = plt.figure(figsize=(8, 4))
    matplotlib.rcParams.update({'font.size': 14})
    plt.suptitle("Power Delay Profile")
    ax = fig.add_subplot()
    # Plot Power Delay Profile and 0-line.
    ax.plot(t, P,
        label="Power Delay Profile $P(\\tau)$", c='k', linestyle='-')
    ax.plot([0, numpy.amax(t)], [0, 0], c='k', lw=0.5)
    # Plot maximum delay.
    ax.scatter(true_maximum_delay, 0,
        label="Maximum delay $T_m$", c='k', marker='x', s=80)
    ax.text(true_maximum_delay, 3*ystr, true_maximum_delay_str)
    # Plot mean delay.
    ax.scatter(true_mean_delay, 0,
        label="Mean delay $\mu_{\\tau}$", c='k', marker='P', s=80)
    ax.text(true_mean_delay, ystr, true_mean_delay_str)
    # Plot RMS delay.
    ax.scatter(true_rms_delay, 0,
        label="RMS delay spread $\sigma_{\\tau}$", c='k', marker='*', s=80)
    ax.text(true_rms_delay, 2*ystr, true_rms_delay_str)
    ax.set_xlabel("$\\tau \ [\mu s]$")
    ax.set_ylabel("$P(\\tau)$")
    ax.set_xlim([0, 0.5])
    ax.set_ylim([-0.5*ymax, 1.1*ymax])
    ax.legend()
    fig.tight_layout(pad=2)

    return Tm, Wc



""" SET UP GP MODEL
"""
class GP(gpytorch.models.ExactGP):
    def __init__(self, x_train, y_train, likelihood):
        """ Takes training data and likelihood and constructs objects neccessary
        for 'forward' module. Commonly mean and kernel module. """
        super(GP, self).__init__(x_train, y_train, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel())# + \
            #gpytorch.kernels.ScaleKernel(gpytorch.kernels.LinearKernel())

    def forward(self, x):
        """ Takes some NxD data and returns 'MultivariateNormal' with prior mean
        and covariance, i.e. mean(x) and NxN covariance matrix K_xx. """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def optimize(self, x, y, likelihood, epochs, lr):
        self.train()
        likelihood.train()
        optimizer = torch.optim.Adam([{'params': self.parameters()}], lr=lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, self)
        for i in range(epochs):
            optimizer.zero_grad()
            output = self(x)
            loss = -mll(output, y)
            loss.backward()
            print(".", end="" , flush=True)
            # print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
            #     i + 1, epochs, loss.item(),
            #     self.covar_module.base_kernel.lengthscale.item(),
            #     self.likelihood.noise.item()))
            optimizer.step()
        print("", flush=True)
        return loss, self.covar_module.base_kernel.lengthscale.item()

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


def get_cov(max_ind, cov, coherence_bound):
    cov_max = cov[max_ind][max_ind]
    # print("cov_max", cov_max)
    cov_find = coherence_bound*cov_max
    # print("cov_find", cov_find)
    curr_ind_r = int(max_ind)
    curr_val_r = float(cov_max)
    # print("curr_val_r", curr_val_r)
    while True:
        if curr_val_r < cov_find:
            break
        curr_ind_r += 1
        curr_val_r = cov[max_ind][curr_ind_r]
    curr_ind_l = int(max_ind)
    curr_val_l = float(cov_max)
    while True:
        if curr_val_l < cov_find:
            break
        curr_ind_l -= 1
        curr_val_l = cov[max_ind][curr_ind_l]
    return curr_ind_l, curr_ind_r


def PredictTransferFunctionGP(range_slices):
    x = numpy.array([])
    y = numpy.array([])
    T_SAMPLES = 52
    x2 = numpy.zeros(T_SAMPLES)
    y2 = numpy.zeros(T_SAMPLES)
    for slice in range_slices:
        t_slice_np = data.get_slice(xy='x', vector=X_np, time=slice, freq=None)
        f = t_slice_np[:,0]
        Y_slice_np = data.get_slice(xy='y', vector=Y_np, time=slice, freq=None)
        fd = (Y_slice_np[:,0] + (1j)*Y_slice_np[:,1])
        abs = numpy.absolute(fd)

        x = numpy.append(x, f)
        y = numpy.append(y, abs)
        x2 = numpy.vstack((x2, f))
        y2 = numpy.vstack((y2, abs))

    # x = numpy.asarray(x)
    # y = numpy.asarray(y)

    # for i in range(len(x)):
    #     x[i] += numpy.random.randn()*0.01
    # sort = x.argsort()
    # x = x[sort]
    # y = y[sort]


    x = numpy.mean(x2[1:,], axis=0)
    y = numpy.mean(y2[1:,], axis=0)

    # x = x2[1:,] # choose which one, x or x2
    # y = y2[1:,]

    # x = x
    # y = y

    f_train = torch.from_numpy(x).reshape(-1,1).type('torch.FloatTensor').contiguous()
    abs_train = torch.from_numpy(y).reshape(-1,).type('torch.FloatTensor').contiguous()

    EPOCHS = 500 # number of iterations during training
    LR = 0.05 # learning rate during training

    gpytorch.settings.skip_posterior_variances(state=False)
    gpytorch.settings.lazily_evaluate_kernels(state=False)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = GP(f_train, abs_train, likelihood)

    PRED_STEPS = 200
    PRED = 26
    loss, ls = model.optimize(f_train, abs_train, likelihood, EPOCHS, LR)
    xpred = torch.linspace(-PRED, PRED, PRED_STEPS)
    mean, var, lower, upper, cov = model.predict(xpred, likelihood)

    COHERENCE_BOUNDS = [0.5, 0.6, 0.7, 0.8, 0.9]
    BOUND = 20
    CW_ALL = []

    x_cw = numpy.linspace(-PRED*(PRED_STEPS/2-BOUND)/(PRED_STEPS/2), PRED*(PRED_STEPS/2-BOUND)/(PRED_STEPS/2), (PRED_STEPS-BOUND)-BOUND)
    fig_cw = plt.figure(figsize=(8, 4))
    plt.suptitle("Coherence bandwidth $W_c$ vs. bound $\\alpha$")
    ax_cw = fig_cw.add_subplot()

    for cb in COHERENCE_BOUNDS:
        cw_all = []
        for i in range(BOUND, PRED_STEPS-BOUND):
            ind_left, ind_right = get_cov(i, cov, cb)
            cw = (ind_right - ind_left) * 312.5
            cw_all.append(cw)

        cw_mean = numpy.mean(cw_all)
        ax_cw.plot(x_cw, cw_all,
            c='k', lw=1)
        cb_str = " " + str(cb)
        ax_cw.text(x_cw[0], cw_all[0]+50, "$\\alpha = $" + str(cb))
        ax_cw.text(x_cw[-1], cw_all[0]+50, "Mean $W_c = $" + str(numpy.around(cw_mean, 1)), ha='right')

    XTICKS = numpy.asarray([-26, -13, 1, 13, 26]) * 312.5 # kHz
    XVALS = [-26, -13, 1, 13, 26]
    ax_cw.set_xticks(XVALS)
    ax_cw.set_xticklabels(XTICKS)
    ax_cw.set_xlabel("$f\ [kHz]$")
    ax_cw.set_ylabel("$W_c \ [kHz]$")
    ax_cw.set_xlim([-PRED, PRED])
    #ax_cw.set_ylim([0, 1.1*numpy.amax(CW_ALL)])
    ax_cw.legend(ncol=3)
    fig_cw.tight_layout(pad=2)
    ax_cw.legend()


    # Plot in figure.
    fig = plt.figure(figsize=(8, 4))
    matplotlib.rcParams.update({'font.size': 14})
    plt.suptitle("GPR of mean of aboslute value (" + str(len(range_slices)) + " sampled transfer functions")
    ax = fig.add_subplot()
    # Plot Power Delay Profile and 0-line.
    ax.scatter(f_train, abs_train,
        label="Training data", c='k', marker='.', alpha=0.025)
    ax.plot(xpred, mean,
        label="Mean", c='k', lw=2)
    ax.fill_between(xpred, lower.numpy(), upper.numpy(),
        alpha=0.2, color='k', label="Variance (95%)")
    ax.text(0, 0+0.05, "Length-scale $\ell = $" + str(numpy.around(ls,5)), ha='center')

    XTICKS = numpy.asarray([-26, -13, 1, 13, 26]) * 312.5 # kHz
    XVALS = [-26, -13, 1, 13, 26]
    ax.set_xticks(XVALS)
    ax.set_xticklabels(XTICKS)
    ax.set_xlabel("$f\ [kHz]$")
    ax.set_ylabel("$|H(f)|$")
    ax.set_xlim([-PRED, PRED])
    ymax = numpy.amax(mean.numpy())
    ax.set_ylim([0, 1.25*ymax])
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

    return loss, ls


def LengthScaleGP(range_slices):
    T_SAMPLES = len(range_slices)

    S = numpy.linspace(1, T_SAMPLES+1, T_SAMPLES)

    EPOCHS = 500 # number of iterations during training
    LR = 0.05 # learning rate during training

    LS = []

    for slice in range_slices:
        t_slice_np = data.get_slice(xy='x', vector=X_np, time=slice, freq=None)
        f = t_slice_np[:,0]
        Y_slice_np = data.get_slice(xy='y', vector=Y_np, time=slice, freq=None)
        fd = (Y_slice_np[:,0] + (1j)*Y_slice_np[:,1])
        abs = numpy.absolute(fd)

        f_train = torch.from_numpy(f).reshape(-1,1).type('torch.FloatTensor').contiguous()
        abs_train = torch.from_numpy(abs).reshape(-1,).type('torch.FloatTensor').contiguous()

        gpytorch.settings.skip_posterior_variances(state=False)
        gpytorch.settings.lazily_evaluate_kernels(state=False)
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = GP(f_train, abs_train, likelihood)

        loss, ls = model.optimize(f_train, abs_train, likelihood, EPOCHS, LR)
        LS.append(ls)

    ls_mean = numpy.mean(LS)
    LS_MEAN = numpy.full(T_SAMPLES, ls_mean)


    matplotlib.rcParams.update({'font.size': 14})
    fig_ls = plt.figure(figsize=(8, 4))
    plt.suptitle("Length-scale vs. sample")
    ax_ls = fig_ls.add_subplot()
    ax_ls.plot(S, LS,
        label="Length-scale", c='k', lw=2)
    ax_ls.plot(S, LS_MEAN,
        label="Mean " + str(numpy.around(ls_mean, 3)), c='k', ls="--", lw=2)
    ax_ls.legend()
    ax_ls.set_xlabel("Sample")
    ax_ls.set_ylabel("Length-scale")
    fig_ls.tight_layout(pad=2)

    return ls_mean, LS



def LengthScaleConstant(range_slices):
    T_SAMPLES = len(range_slices)

    S = numpy.linspace(1, T_SAMPLES+1, T_SAMPLES)

    EPOCHS = 500 # number of iterations during training
    LR = 0.05 # learning rate during training

    MAX_D = []
    MEAN_D = []
    RMS_D = []
    LS = []

    for slice in range_slices:
        curr_slice = range(slice, slice+1)
        F_ABS_MEAN, F_REAL_MEAN, F_IMAG_MEAN, T_ABS_MEAN, T_REAL_MEAN, T_IMAG_MEAN, t = ImpulseResponseAverage(curr_slice)
        P = PowerDelayProfile(T_ABS_MEAN)
        MAX_DELAY = MaximumDelay(T_ABS_MEAN)
        MEAN_DELAY = MeanDelay(T_ABS_MEAN)
        RMS_DELAY = RMSDelay(T_ABS_MEAN)

        T_MAX = numpy.amax(t)
        T_NUM = len(t)

        # Compute true delay times.
        true_maximum_delay = T_MAX * MAX_DELAY/T_NUM # [microseconds]
        true_mean_delay = T_MAX * MEAN_DELAY/T_NUM # [microseconds]
        true_rms_delay = T_MAX * RMS_DELAY/T_NUM # [microseconds]

        MAX_D.append(true_maximum_delay)
        MEAN_D.append(true_mean_delay)
        RMS_D.append(true_rms_delay)

        t_slice_np = data.get_slice(xy='x', vector=X_np, time=slice, freq=None)
        f = t_slice_np[:,0]
        Y_slice_np = data.get_slice(xy='y', vector=Y_np, time=slice, freq=None)
        fd = (Y_slice_np[:,0] + (1j)*Y_slice_np[:,1])
        abs = numpy.absolute(fd)

        f_train = torch.from_numpy(f).reshape(-1,1).type('torch.FloatTensor').contiguous()
        abs_train = torch.from_numpy(abs).reshape(-1,).type('torch.FloatTensor').contiguous()

        gpytorch.settings.skip_posterior_variances(state=False)
        gpytorch.settings.lazily_evaluate_kernels(state=False)
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = GP(f_train, abs_train, likelihood)

        loss, ls = model.optimize(f_train, abs_train, likelihood, EPOCHS, LR)
        LS.append(ls)

    C_MAX = numpy.divide(LS, MAX_D)
    C_MEAN = numpy.divide(LS, MEAN_D)
    C_RMS = numpy.divide(LS, RMS_D)

    matplotlib.rcParams.update({'font.size': 14})
    fig = plt.figure(figsize=(8, 8))
    plt.suptitle("Length-scale/delay spread vs. samples")
    ax1 = fig.add_subplot(3, 1, 1)
    ax1.plot(C_MAX,
        c='k', lw=2)
    ax2 = fig.add_subplot(3, 1, 2)
    ax2.plot(C_MEAN,
        c='k', lw=2)
    ax2 = fig.add_subplot(3, 1, 3)
    ax2.plot(C_RMS,
        c='k', lw=2)


    fig.tight_layout(pad=2)





""" RUN """
RANGE_SLICES = range(1,51)
SLICE = 5

# test_ifft()
ImpulseResponseAverage(RANGE_SLICES, plot=True)
#Tm, Wc = PowerDelayProfileAverage(RANGE_SLICES)
#LOSS, LS1 = PredictTransferFunctionGP(RANGE_SLICES)
#LS_MEAN, LS2 = LengthScaleGP(RANGE_SLICES)
#CoherenceBandwidthGP(SLICE)
#AutoCorrelationAbs(RANGE_SLICES)
LengthScaleConstant(RANGE_SLICES)

# print("Power Delay Profile,            Tm = ", Tm)
# print("Power Delay Profile, Wc = 1/(2*Tm) = ", Wc)
# print("GP Predict Length-scale,       LS1 = ", LS1)
# print("GP Length-scale,           LS_MEAN = ", LS_MEAN)
# print("Constant:                        k = ", Wc/LS_MEAN)

plt.show()


""" RESULTS
00
Power Delay Profile,            Tm =  0.24615384615384617
Power Delay Profile, Wc = 1/(2*Tm) =  2031.25
GP Predict Length-scale,       LS1 =  2.935702323913574
GP Length-scale,           LS_MEAN =  3.122010293006897
Constant:                        k =  650.6224545607264

01
Power Delay Profile,            Tm =  0.12307692307692308
Power Delay Profile, Wc = 1/(2*Tm) =  4062.5
GP Predict Length-scale,       LS1 =  6.781728744506836
GP Length-scale,           LS_MEAN =  10.06928087234497
Constant:                        k =  403.45482974435197

02


"""

CONST = []

def results():
    fig = plt.figure(figsize=(8, 4))
    plt.suptitle("Proportional constant for different data")
    ax = fig.add_subplot()
    ax.plt(CONST)








def CoherenceBandwidthGP(slice):
    t_slice_np = data.get_slice(xy='x', vector=X_np, time=slice, freq=None)
    f = t_slice_np[:,0]
    Y_slice_np = data.get_slice(xy='y', vector=Y_np, time=slice, freq=None)
    fd = (Y_slice_np[:,0] + (1j)*Y_slice_np[:,1])
    abs = numpy.absolute(fd)

    f_train = torch.from_numpy(f).reshape(-1,1).type('torch.FloatTensor').contiguous()
    abs_train = torch.from_numpy(abs).type('torch.FloatTensor').contiguous()

    EPOCHS = 400 # number of iterations during training
    LR = 0.05 # learning rate during training

    gpytorch.settings.skip_posterior_variances(state=False)
    gpytorch.settings.lazily_evaluate_kernels(state=False)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = GP(f_train, abs_train, likelihood)

    PRED = 26
    PRED_STEPS = 200


    model.optimize(f_train, abs_train, likelihood, EPOCHS, LR)
    xpred = torch.linspace(-PRED, PRED, PRED_STEPS)
    mean, var, lower, upper, cov = model.predict(xpred, likelihood)

    peak, _ = scipy.signal.find_peaks(mean)
    COHERENCE_BOUND = 0.5
    ind_left, ind_right = get_cov(peak[1], cov, COHERENCE_BOUND)
    print(ind_left, ind_right)
    cw = (ind_right - ind_left) * 312.5
    print("Coherence bandwidth:", cw, "kHz")

    # Plot in figure.
    fig = plt.figure(figsize=(8, 4))
    matplotlib.rcParams.update({'font.size': 14})
    plt.suptitle("Training data")
    ax = fig.add_subplot()
    # Plot Power Delay Profile and 0-line.
    ax.scatter(f_train, abs_train,
        label="Training data", c='k', marker='x', alpha=0.5)
    ax.plot(xpred, mean,
        label="Mean", c='k', lw=2)
    ax.fill_between(xpred, lower.numpy(), upper.numpy(),
        alpha=0.2, color='k', label="Variance (95%)")
    XTICKS = numpy.asarray([-26, -13, 1, 13, 26]) * 312.5 # kHz
    XVALS = [-26, -13, 1, 13, 26]
    ax.set_xticks(XVALS)
    ax.set_xticklabels(XTICKS)
    ax.set_xlabel("$f\ [kHz]$")
    ax.set_xlim([-PRED, PRED])
    ax.set_ylim([0, 8])
    ax.legend(ncol=3)
    fig.tight_layout(pad=2)

    fig_cov = plt.figure(figsize=(5, 5))
    plt.suptitle("Posterior covariance matrix")
    ax_cov = fig_cov.add_subplot()
    cmap = 'jet'
    ax_cov.pcolormesh(cov.detach().numpy(), cmap=cmap)
    plt.gca().invert_yaxis()





def AutoCorrelationAbs(range_slices):
    F_ABS_MEAN, F_REAL_MEAN, F_IMAG_MEAN, T_ABS_MEAN, T_REAL_MEAN, T_IMAG_MEAN, t = ImpulseResponseAverage(range_slices)
    result = numpy.correlate(F_ABS_MEAN, F_ABS_MEAN, mode='full')
    result = result[result.size//2:]/numpy.amax(result)
    print(F_ABS_MEAN.shape)
    print(result.shape)
    print(result)

    COHERENCE_BOUND = 0.5

    ind = 0
    value = 0
    for i in range(len(result)):
        if result[i] < COHERENCE_BOUND:
            ind = i
            value = result[i]
            break
    freq = ind * 312.5

    matplotlib.rcParams.update({'font.size': 14})
    fig_corr = plt.figure(figsize=(8,6))
    plt.suptitle("Autocorrelation of absolute value mean")
    ax_corr = fig_corr.add_subplot(1, 1, 1)
    ax_corr.plot(result, color="k", alpha=1, lw=2, linestyle='--', label="Autocorrelation")
    ax_corr.plot([0, result.size], [COHERENCE_BOUND, COHERENCE_BOUND], color='r', alpha=0.5)
    ax_corr.scatter([ind], [value], c='k')
    ax_corr.text(ind, value, freq)
    XTICKS = numpy.asarray([1, 14, 27, 40, 53]) * 312.5 # kHz
    XVALS = [1, 14, 27, 40, 53]
    ax_corr.set_xticks(XVALS)
    ax_corr.set_xticklabels(XTICKS)
    ax_corr.set_xlabel("$\Delta f\ [kHz]$")
    ax_corr.set_ylabel("Normalized autocorrelation of $|H(f)|$")
    ax_corr.set_xlim([0, result.size])



























"""      OLD BELOW   OLD BELOW    OLD BELOW    OLD BELOW    OLD BELOW        """



""" SPLIT DATA INTO TRAINING/TEST DATA
    Put data into training and test data. Convert to torch tensors.
"""
T_SLICE = 5 # what time slice of frequncies to get from the imported data

X_np = data.get_x() # get imported input points
Y_np = data.get_y() # get imported output points
Y_abs_np = data.get_y_abs() # get absolute value, not really used except for plottin in 3D

# Get slice in time, i.e. array of freqeuncies over that time slice.
t_slice_np = data.get_slice(xy='x', vector=X_np, time=T_SLICE, freq=None)
Y_slice_np = data.get_slice(xy='y', vector=Y_np, time=T_SLICE, freq=None)
# Convert real part a and imaginary part b into a 52x1 complex vector.
Y_slice_complex_np = (Y_slice_np[:,0] + (1j)*Y_slice_np[:,1])
# Compute absolute value of slice.
abs_slice_np = numpy.absolute(Y_slice_complex_np)

# Convert to tensor for training of GP.
TRAIN_NUM = 1
f_train = torch.from_numpy(t_slice_np[:,0].reshape(-1,1)).type('torch.FloatTensor').contiguous()[0::TRAIN_NUM]
abs_train = torch.from_numpy(abs_slice_np).type('torch.FloatTensor').contiguous()[0::TRAIN_NUM]








""" FOURIER TRANSFORM FOR SPREAD DELAY
    Compute IFFT of transfer function and get absolute value, real and imaginary part.
    Needs to be "rolled", so that center frequency is in the beginning.
"""
# Compute IFFT of transfer function and get absolute value, real and imaginary part.
Y_slice_complex_shifted_np = numpy.roll(Y_slice_complex_np, shift=26) # shift transfer function
Y_ifft = numpy.fft.ifft(Y_slice_complex_shifted_np) # compute IFFT
Y_ifft = numpy.insert(Y_ifft, 0, 0) # insert 0 at element 0, since discrete, first pulse starts at index 1
Y_ifft_abs = numpy.absolute(Y_ifft) # compute aboslute value
Y_ifft_a = Y_ifft.real # take out real part
Y_ifft_b = Y_ifft.imag # take out imaginary part





a = 0.1 # the boundary for energy to fall to, a * tot_energy_in_power_delay_profile

# Compute delays.
P = PowerDelayProfile(Y_ifft, norm=True) # compyte PDP, take away noise
maximum_delay = MaximumDelay(Y_ifft, a)
mean_delay = MeanDelay(Y_ifft)
rms_delay = RMSDelay(Y_ifft)

# True sampling period.
Ts = 1/(53*312.5*1000) # [1/(N*kHz*1000) = seconds], N = 53 bins of frequencies

# Compute true delay times.
true_maximum_delay = maximum_delay * Ts * 1000000 # [microseconds]
true_mean_delay = mean_delay * Ts * 1000000 # [microseconds]
true_rms_delay = rms_delay * Ts * 1000000 # [microseconds]

# Prepare string to print in plot.
true_maximum_delay_str = "$T_m = $" + str(numpy.around(true_maximum_delay, 5)) + " $[\mu s]$"
true_mean_delay_str = "$\mu_{\\tau} = $" + str(numpy.around(true_mean_delay, 5)) + " $[\mu s]$"
true_rms_delay_str = "$\sigma_{\\tau} = $" + str(numpy.around(true_rms_delay, 5)) + " $[\mu s]$"

# Plot in figure.
fig = plt.figure(figsize=(8, 5))
matplotlib.rcParams.update({'font.size': 14})
plt.suptitle("Power Delay Profile - $\mathcal{IFFT}\{ P( \\tau ) = |h(\\tau)|^2 \}$")
ax = fig.add_subplot()
# Plot Power Delay Profile and 0-line.
ax.plot(P,
    label="Power Delay Profile $P(\\tau)$", c='k', linestyle='-')
ax.plot([0, 50], [0, 0], c='k', lw=0.5)
# Plot maximum delay.
ax.scatter(maximum_delay, 0,
    label="Maximum delay $T_m$", c='k', marker='x', s=80)
ax.text(maximum_delay, -2.6, true_maximum_delay_str)
# Plot mean delay.
ax.scatter(mean_delay, 0,
    label="Mean delay $\mu_{\\tau}$", c='k', marker='P', s=80)
ax.text(mean_delay, -1.8, true_mean_delay_str)
# Plot RMS delay.
ax.scatter(rms_delay, 0,
    label="RMS delay spread $\sigma_{\\tau}$", c='k', marker='*', s=80)
ax.text(rms_delay, -1.0, true_rms_delay_str)

# Plot true time on x-axis.
TIME_STEPS = 20
true_time_plot = numpy.linspace(0, 52*Ts*1000000, TIME_STEPS) # [microseconds]
ax.set_xticks(numpy.linspace(0, 52, TIME_STEPS))
ax.set_xticklabels(numpy.around(true_time_plot, decimals=3))

# Plot info and set axes.
ax.legend()
ax.set_xlabel("$t \ [\mu s]$")
ax.set_xlim([0, 15])
ax.set_ylim([-3, 10])
fig.tight_layout(pad=3.0)



""" FOURIER TRANSFORM OF POWE DELAY PROFILE
    Compute FFT of Power Delay Profile in order to estimate coherence bandwidth.
"""
def FFTP(P, a):
    """ Compute Fourier Transform of Power Delay Profile.
    """
    P = P/numpy.sum(P)
    FFTP = numpy.fft.fft(P)
    #FFTP = numpy.roll(FFTP, shift=FFTP.size//2)
    abs = numpy.absolute(FFTP)
    FFTP = abs
    start = FFTP[0]
    find = a * start
    CW = None
    for f in range(len(FFTP)):
        if FFTP[f] <= find:
            CW = f
            break
    return FFTP, CW


a_fft = 0.5
FFTP, cw_ind = FFTP(P, a_fft)
true_cw = cw_ind/52 * 312.5 * 53

# Plot.
fig2 = plt.figure(figsize=(8, 5))
plt.suptitle("Inverse Fourier Transform of $P(tau)$")
ax2 = fig2.add_subplot()
ax2.plot(FFTP,
    c='k', label="$\mathcal{F}\{ P(tau) \}")
ax2.scatter(cw_ind, FFTP[cw_ind],
    label="Coherence bandwidth$", c='k', marker='*', s=80)

true_cw_str = "$W_c = $" + str(numpy.around(true_cw, 5)) + " $[kHz]$"
ax2.text(cw_ind+1, FFTP[cw_ind], true_cw_str)



""" PLOT ABSOLUTE VALUE TRAINING POINTS
"""
fig_data_f = plt.figure(figsize=(8,6))
plt.suptitle("Coherence bandwidth using GPR")
ax_data_f = fig_data_f.add_subplot(1, 1, 1)
ax_data_f.scatter(f_train, abs_train,
    label="Training points", marker="x", color="k", alpha=0.75)
ax_data_f.plot(f_train, abs_train, color="k", alpha=0.1, lw=4)



""" SET UP GP MODEL
"""
class GP(gpytorch.models.ExactGP):
    def __init__(self, x_train, y_train, likelihood):
        """ Takes training data and likelihood and constructs objects neccessary
        for 'forward' module. Commonly mean and kernel module. """
        super(GP, self).__init__(x_train, y_train, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel())#+ gpytorch.kernels.LinearKernel()# + \
            #gpytorch.kernels.PeriodicKernel()

    def forward(self, x):
        """ Takes some NxD data and returns 'MultivariateNormal' with prior mean
        and covariance, i.e. mean(x) and NxN covariance matrix K_xx. """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)



""" TRAIN AND OPTIMIZE
"""
NUM_EPOCHS = 500 # number of iterations during training
LEARNING_RATE = 0.02 # learning rate during training

gpytorch.settings.skip_posterior_variances(state=False)
gpytorch.settings.lazily_evaluate_kernels(state=False)
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = GP(f_train, abs_train, likelihood)

model.train()
likelihood.train()
optimizer = torch.optim.Adam([{'params': model.parameters()}], lr=LEARNING_RATE)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

for i in range(NUM_EPOCHS):
    optimizer.zero_grad()
    output = model(f_train)
    loss = -mll(output, abs_train)
    loss.backward()
    print('Iter %d/%d - Loss: %.3f' % (i + 1, NUM_EPOCHS, loss.item()))
    optimizer.step()



""" PREDICTIONS AND PLOTTING
    Predict for both time and frequency and plot for real and imaginary part
    respectively.
"""
PRED_STEPS = 250 # number of steps/points to predict
PRED_START = -26 # point to start prediction from
PRED_END = 26 # point to end prediction with

model.eval()
likelihood.eval()

with torch.no_grad():
    pred = torch.linspace(PRED_START, PRED_END, PRED_STEPS)
    model_out = model(pred) # returns (predictive_mean, predictive_covar)
    model_out.covariance_matrix
    cov_mat = model_out.covariance_matrix
    # predictions = likelihood(model_out)
    predictions = model_out
    mean = predictions.mean
    var = predictions.variance
    lower, upper = predictions.confidence_region()



""" COMPUTE AUTO-CORRELATION OF MEAN FROM GP
"""
def autocorr_norm(x):
    result = numpy.correlate(x, x, mode='full')
    return result[result.size//2:]/numpy.amax(result)

COHERENCE_BOUND = 0.9

autocorr_mean = autocorr_norm(mean)
peaks_autocorr, _ = scipy.signal.find_peaks(autocorr_mean)
curr = 0
autocorr_curr = autocorr_mean[curr]
# Find coherence bandwidth.
while True:
    if autocorr_curr < COHERENCE_BOUND:
        break
    curr += 1
    autocorr_curr = autocorr_mean[curr]

fig_corr = plt.figure(figsize=(8,6))
plt.suptitle("Autocorrelation of GP mean")
ax_corr = fig_corr.add_subplot(1, 1, 1)
ax_corr.plot(autocorr_mean, color="k", alpha=1, lw=2, linestyle='--', label="Autocorrelation")
ax_corr.plot([0, PRED_STEPS], [COHERENCE_BOUND, COHERENCE_BOUND], color='k', alpha=1, lw=1, label="Bound")
ax_corr.scatter(curr, autocorr_mean[curr], color='k', marker="*", s=80)
ax_corr.plot([curr, curr], [0, autocorr_mean[curr]], color='k', alpha=1, lw=0.5, linestyle="dotted")
ax_corr.plot([0, curr], [0, 0], color='k', alpha=1, lw=3, label="Coherence bandwidth")

ax_corr.set_xticks(numpy.linspace(0, 49*PRED_STEPS/53, 7))
corr_freqs = (data.get_frequencies()-data.get_frequencies()[0])[0::8]
ax_corr.set_xticklabels(corr_freqs)

corr_Wc = curr/PRED_STEPS*315.5*53
str_corr_Wc = "$W_c = $" + str(numpy.around(corr_Wc, decimals=1)) + " $[kHz]$"
ax_corr.text(curr, -0.1, str_corr_Wc)

ax_corr.set_xlabel("$\Delta f\ [kHz]$")
ax_corr.set_xlim([0, PRED_STEPS])
ax_corr.set_ylim([-0.25, 1.25])
ax_corr.legend()



""" COMPUTE AUTO-CORRELATION OF TRANSFER FUNCTION
"""
def autocorr_c_norm(x):
    result = numpy.correlate(numpy.conjugate(x), x, mode='full')
    return result[result.size//2:]

COHERENCE_BOUND = 0.9

autocorr_c = autocorr_c_norm(Y_slice_complex_np)
autocorr_c_abs = numpy.absolute(autocorr_c)
autocorr_c_abs = autocorr_c_abs/numpy.amax(autocorr_c_abs)
peaks_autocorr_c, _ = scipy.signal.find_peaks(autocorr_c_abs)
curr_c = 0
autocorr_curr_c = autocorr_c_abs[curr_c]
# Find coherence bandwidth.
while True:
    if autocorr_curr_c < COHERENCE_BOUND:
        break
    curr_c += 1
    autocorr_curr_c = autocorr_c_abs[curr_c]

fig_corr_c = plt.figure(figsize=(8,6))
plt.suptitle("Autocorrelation of transfer function")
ax_corr_c = fig_corr_c.add_subplot(1, 1, 1)
ax_corr_c.plot(autocorr_c_abs, color="k", alpha=1, lw=2, linestyle='--', label="Autocorrelation")
ax_corr_c.plot([0, len(autocorr_c_abs)], [COHERENCE_BOUND, COHERENCE_BOUND], color='k', alpha=1, lw=1, label="Bound")
ax_corr_c.scatter(curr_c, autocorr_c_abs[curr_c], color='k', marker="*", s=80)
ax_corr_c.plot([curr_c, curr_c], [0, autocorr_c_abs[curr_c]], color='k', alpha=1, lw=0.5, linestyle="dotted")
ax_corr_c.plot([0, curr_c], [0, 0], color='k', alpha=1, lw=3, label="Coherence bandwidth")

ax_corr_c.set_xticks(numpy.linspace(0, 49/53*52, 7))
corr_c_freqs = (data.get_frequencies()-data.get_frequencies()[0])[0::8]
ax_corr_c.set_xticklabels(corr_c_freqs)

corr_c_Wc = curr_c/52*315.5*53
str_corr_Wc = "$W_c = $" + str(numpy.around(corr_c_Wc, decimals=1)) + " $[kHz]$"
ax_corr_c.text(curr_c, -0.1, str_corr_Wc)

ax_corr_c.set_xlabel("$\Delta f\ [kHz]$")
ax_corr_c.set_xlim([0, len(autocorr_c_abs)])
ax_corr_c.set_ylim([-0.25, 1.25])
ax_corr_c.legend()



""" PLOT PREDICTIONS AND COVARIANCE MATRIX
"""
ax_data_f.plot(pred.numpy(), mean.numpy(),
    color='k', lw=2, label="Mean", alpha=1, linestyle='--')
ax_data_f.fill_between(pred.numpy(), lower.numpy(), upper.numpy(),
    alpha=0.2, color='k', label="Variance (95%)")


cov = cov_mat

# CALCULATE MEAN VARIANCE OF DIAGONAL AND SET DIAGONAL TO 1
sum = 0
for i in range(PRED_STEPS):
    sum += cov[i][i]
mean_diag = sum/PRED_STEPS
# print("mean_diag:", mean_diag)
cov = cov/mean_diag



""" FIND COHERENCE BANDWIDTH FOR PEAK
"""
torch.set_printoptions(precision=20)

COHERENCE_BOUND = 0.5

def get_cov(max_ind, cov):
    cov_max = cov[max_ind][max_ind]
    # print("cov_max", cov_max)
    cov_find = COHERENCE_BOUND*cov_max
    # print("cov_find", cov_find)
    curr_ind_r = int(max_ind)
    curr_val_r = float(cov_max)
    # print("curr_val_r", curr_val_r)
    while True:
        if curr_val_r < cov_find:
            break
        curr_ind_r += 1
        curr_val_r = cov[max_ind][curr_ind_r]
    curr_ind_l = int(max_ind)
    curr_val_l = float(cov_max)
    while True:
        if curr_val_l < cov_find:
            break
        curr_ind_l -= 1
        curr_val_l = cov[max_ind][curr_ind_l]
    return curr_ind_l, curr_ind_r

peaks, _ = scipy.signal.find_peaks(mean)
# print("peaks found:", peaks)
max_ind_1 = peaks[1]
# max_ind_2 = peaks[2]

max_f_1 = pred[max_ind_1]
max_val_1 = mean[max_ind_1]
ind_l_1, ind_r_1 = get_cov(max_ind_1, cov)

# max_f_2 = pred[max_ind_2]
# max_val_2 = mean[max_ind_2]
# ind_l_2, ind_r_2 = get_cov(max_ind_2, cov)

# CALCULATE COHERENCE BANDWIDTH
diff_ind_1 = ind_r_1 - ind_l_1
rel_pred_1 = diff_ind_1/PRED_STEPS
Wc_gp_1 = rel_pred_1*53*312.5 #kHz

# diff_ind_2 = ind_r_2 - ind_l_2
# rel_pred_2 = diff_ind_2/PRED_STEPS
# Wc_gp_2 = rel_pred_2*53*312.5 #kHz



""" PLOT COHERENCE BANDWIDTH
"""
bw_f_l_1 = pred[ind_l_1]
bw_f_r_1 = pred[ind_r_1]
bw_abs_l_1 = mean[ind_l_1]
bw_abs_r_1 = mean[ind_r_1]

# bw_f_l_2 = pred[ind_l_2]
# bw_f_r_2 = pred[ind_r_2]
# bw_abs_l_2 = mean[ind_l_2]
# bw_abs_r_2 = mean[ind_r_2]

ax_data_f.scatter(max_f_1, max_val_1, color='k', marker="*", label="Maximum")
# ax_data_f.scatter(max_f_2, max_val_2, color='k', marker="*")

ax_data_f.plot([bw_f_l_1, bw_f_r_1], [0, 0],
    c='k', lw='3', label="Coherence bandwidth")
# ax_data_f.plot([bw_f_l_2, bw_f_r_2], [bw_abs_l_2, bw_abs_r_2],
#     c='k', lw='3')

ax_data_f.plot([max_f_1, max_f_1], [max_val_1, 0],
    c='k', lw='0.5')
ax_data_f.plot([bw_f_l_1, bw_f_l_1], [bw_abs_l_1, 0],
    c='k', lw='0.5')
ax_data_f.plot([bw_f_r_1, bw_f_r_1], [bw_abs_r_1, 0],
    c='k', lw='0.5')

str_Wc_gp1 = "$W_c = $" + str(numpy.around(Wc_gp_1, decimals=1)) + " $[kHz]$"
ax_data_f.text(pred[max_ind_1], -0.4, str_Wc_gp1, ha='center')

# ax_data_f.plot([max_f_2, max_f_2], [max_val_2, 0],
#     c='k', lw='0.5')
# ax_data_f.plot([bw_f_l_2, bw_f_l_2], [bw_abs_l_2, 0],
#     c='k', lw='0.5')
# ax_data_f.plot([bw_f_r_2, bw_f_r_2], [bw_abs_r_2, 0],
#     c='k', lw='0.5')
#
# str_Wc_gp2 = "$W_c = $" + str(numpy.around(Wc_gp_2, decimals=1)) + " $[kHz]$"
# ax_data_f.text(pred[max_ind_2], -0.4, str_Wc_gp2, ha='center')

ax_data_f.set_xticks(numpy.linspace(-26, int(data.get_freq()[46]), 5))
ax_data_f.set_xticklabels(data.get_frequencies()[0::12])
ax_data_f.set_xlim([PRED_START, PRED_END])
ax_data_f.set_ylim([-0.8, 7])



""" PLOT COVARIANCE MATRIX
"""
fig_cov = plt.figure(figsize=(5, 5))
cmap = 'jet'
ax_cov_yy = fig_cov.add_subplot()
ax_cov_yy.set_title("Posterior covariance matrix")
ax_cov_yy.pcolormesh(cov.detach().numpy(), cmap=cmap)

ax_cov_yy.plot([max_ind_1, max_ind_1], [ind_l_1, ind_r_1], c='w', lw=2)
ax_cov_yy.plot([ind_l_1, ind_r_1], [max_ind_1, max_ind_1], c='w', lw=2)

# ax_cov_yy.plot([max_ind_2, max_ind_2], [ind_l_2, ind_r_2], c='w', lw=2)
# ax_cov_yy.plot([ind_l_2, ind_r_2], [max_ind_2, max_ind_2], c='w', lw=2)

plt.gca().invert_yaxis()
plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[]);



""" PLOT ALL FIGURES
"""
# ax_data.legend()
ax_data_f.set_xlabel("$f\ [kHz]$")
ax_data_f.set_ylabel("$|H(f)|$")
# ax_data_f.set_xlim([PRED_START, PRED_END])
# ax_data_f.set_ylim([0, 6])
ax_data_f.legend()
plt.show()
