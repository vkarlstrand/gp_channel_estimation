""" Data.py

    Import WIFI capture data from .csv files on the form:

    TIME, SSID, MAC, SEQ, SNR, CFO, CSI[...]

    TIME:   Time stamp                      2019-03-31 14:56:27.198
    SSID:   SSID                            No SSID
    MAC:    MAC adress                      "ff:ff:ff:ff:ff:ff"
    SEQ:    Packet sequence number          3701
    SNR:    Signal to Noise Ratio           23
    CFO:    Carrier Freqeuency Offset ghm   5483
    CSI:    Channel State Information       0.759348-0.209251i, ..., 0.848393-0.316414i

    CSI consists of 52 samples over 52 frequencies for each time stamp.

"""



import numpy as np
import csv
from datetime import datetime
from datetime import timedelta
from datetime import time
import torch
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


""" HELPER FUNCTIONS
"""
# Used to normalize data.

def normalize(arr):
    # Normalize to (-1,1)
    amin = np.amin(arr)
    arr = arr - amin
    amax = np.amax(arr)
    arr = 2 * arr/amax - 1
    return arr, amin, amax

def get_mesh_x(tensor, f_num, t_num):
    f_col = tensor[:,0]
    f = f_col[0:f_num].numpy()
    t_col = tensor[:,1]
    t = np.array([])
    for i in range(t_num):
        t = np.append(t, t_col[i*f_num])
    mesh_f, mesh_t = np.meshgrid(f, t)
    # print(mesh_f, mesh_t)
    return mesh_f, mesh_t

def get_mesh_y(tensor, f_num, t_num):
    tensor = tensor.numpy().reshape(t_num, f_num)
    # print(tensor)
    return tensor

def plot3D(data, title, tslice=None, fslice=None):
    X = torch.from_numpy(data.get_x()).type('torch.FloatTensor').contiguous()
    Ya = torch.from_numpy(data.get_y()[:,0]).type('torch.FloatTensor').contiguous()
    Yb = torch.from_numpy(data.get_y()[:,1]).type('torch.FloatTensor').contiguous()
    Yabs = torch.from_numpy(data.get_y_abs_ang()[:,0]).type('torch.FloatTensor').contiguous()
    Yang = torch.from_numpy(data.get_y_abs_ang()[:,1]).type('torch.FloatTensor').contiguous()
    f, t = get_mesh_x(tensor=X, f_num=data.get_f_num(), t_num=data.get_t_num())
    print(t)
    # f = data.get_x()[:,0].reshape(data.get_f_num(), data.get_t_num())
    # t = data.get_x()[:,1].reshape(data.get_f_num(), data.get_t_num())
    # a = data.get_y()[:,0].reshape(data.get_f_num(), data.get_t_num())
    # b = data.get_y()[:,1].reshape(data.get_f_num(), data.get_t_num())
    a = get_mesh_y(tensor=Ya, f_num=data.get_f_num(), t_num=data.get_t_num())
    b = get_mesh_y(tensor=Yb, f_num=data.get_f_num(), t_num=data.get_t_num())
    # abs = data.get_y_abs_ang()[:,0].reshape(data.get_f_num(), data.get_t_num())
    # ang = data.get_y_abs_ang()[:,1].reshape(data.get_f_num(), data.get_t_num())
    abs = get_mesh_y(tensor=Yabs, f_num=data.get_f_num(), t_num=data.get_t_num())
    ang = get_mesh_y(tensor=Yang, f_num=data.get_f_num(), t_num=data.get_t_num())

    fig = plt.figure(constrained_layout=True, figsize=(10,8))
    fig.suptitle(title)
    spec = gridspec.GridSpec(ncols=2, nrows=2, figure=fig)

    cmap = 'jet'
    ax1 = fig.add_subplot(spec[0, 0], projection='3d')
    ax1.plot_surface(f, t, a, cmap=cmap, alpha=0.5)
    ax1.set_xlabel("$f \ [MHz]$")
    ax1.set_ylabel("$t \ [s]$")
    ax1.set_zlabel("real $H(f,t)|$")
    ax2 = fig.add_subplot(spec[0, 1], projection='3d')
    ax2.plot_surface(f, t, b, cmap=cmap, alpha=0.5)
    ax2.set_xlabel("$f \ [MHz]$")
    ax2.set_ylabel("$t \ [s]$")
    ax2.set_zlabel("imag $H(f,t)|$")
    ax3 = fig.add_subplot(spec[1, 0], projection='3d')
    ax3.plot_surface(f, t, abs, cmap=cmap, alpha=0.5)
    ax3.set_xlabel("$f \ [MHz]$")
    ax3.set_ylabel("$t \ [s]$")
    ax3.set_zlabel("abs $ H(f,t)$")
    ax4 = fig.add_subplot(spec[1, 1], projection='3d')
    ax4.plot_surface(f, t, ang, cmap=cmap, alpha=0.5)
    ax4.set_xlabel("$f \ [MHz]$")
    ax4.set_ylabel("$t \ [s]$")
    ax4.set_zlabel("ang $ H(f,t)$")




""" Data
    Imports, load and prepare data.
"""
class Data:
    def __init__(self, name=None):
        self.name = name
        self.file_name = None
        # Data from csv file
        self.time = []          # time stamp
        self.last_time = None
        self.ssid = []          # ssid
        self.mac = []           # mac adress
        self.seq = []           # packet sequence number
        self.snr = []           # signal to noise ratio
        self.cfo = []           # carrier frequency offset ghm
        self.csi = []           # channsel state information (52 complex numbers)
        # Frequencies
        self.freq = np.array([-26, -25, -24, -23, -22, -21, -20, -19, -18, -17, -16,
                            -15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4,
                            -3, -2, -1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                            14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]).astype(float)
        #print(self.freq.shape)
        #self.freq = list(range(1, 53))  # 52 frequencies: [1,...,52], temporary values
        # Arrays for computation
        self.x = None           # input vector [[f_1 t_1] ... [f_n t_n] ... [f_N t_N]]^T
        self.y = None           # output vector [[z_1] ... [z_n] ... [z_N]]^T, where z_n = a_n +i*b_n
        self.y_abs = None       # output vector [[|z_1|] ... [|z_n|] ... [|z_N|]]^T
        self.y_ang = None       # output vector [[ang z_1] ... [ angz_n] ... [ang z_N]]^T
        self.y_complex = None   # holds complex values a+jb
        self.y_abs_ang = None

        self.norms = []         # save normalizing constant to unnormalize again

        # Information
        self.t_num = None       # number of time points imported
        self.f_num = None       # number of frequencies (52)

        self.frequencies = np.array([-26, -25, -24, -23, -22, -21, -20, -19, -18, -17, -16,
                            -15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4,
                            -3, -2, -1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                            14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]).astype(float) * 312.5

    # Getters
    def get_x(self):
        return self.x
    def get_y(self):
        return self.y
    def get_y_a(self):
        return self.y[:,0]
    def get_y_b(self):
        return self.y[:,1]
    def get_y_abs(self):
        return self.y_abs
    def get_y_ang(self):
        return self.y_ang
    def get_y_complex(self):
        return self.y_complex
    def get_y_abs_ang(self):
        return self.y_abs_ang

    def get_time(self):
        return self.time
    def get_last_time(self):
        return self.last_time
    def get_freq(self):
        return self.freq
    def get_frequencies(self):
        return self.frequencies
    def get_t_num(self):
        return self.t_num
    def get_f_num(self):
        return self.f_num
    def get_mac(self):
        return self.mac


    def load(self, csv_file, mac_str, limit=None, skip=None, jump=None, skip_close=False, mac_all=False):
        """ Reads a csv file and parses data into arrays. """
        limit_str = ""
        skip_str = ""
        if limit != None:
            limit_str = " with a limit of " + str(limit) + " time stamps"
        if skip != None:
            skip_str = " and skipping " + str(skip)
        print("Starting to load data from '" + csv_file + "'" + limit_str + skip_str + "... ", end='')
        self.file_name = csv_file
        with open(csv_file, 'r') as file:
            csv_reader = csv.reader(file)
            limit_counter = 0
            skip_counter = 0
            jump_counter = 0
            skip_close_prev_time = 0
            for line in csv_reader:
                if limit != None and limit_counter >= limit:
                    break
                elif skip_counter < skip:
                    skip_counter += 1
                    pass
                elif jump != None and jump_counter < jump:
                    jump_counter += 1
                else:
                    if (line[2] == mac_str or mac_all == True):
                        skip_close_curr_time = self.datetime_to_sec(line[0])
                        if skip_close and ((skip_close_curr_time - skip_close_prev_time) < skip_close): # 0.25
                            pass
                        else:
                            self.time.append(skip_close_curr_time)
                            self.ssid.append(line[1])
                            self.mac.append(line[2])
                            self.seq.append(int(line[3]))
                            self.snr.append(int(line[4]))
                            self.cfo.append(int(line[5]))
                            csi = []
                            for complex_str in line[6:]:
                                complex_str = complex_str.replace("i", "j")
                                complex_num = complex(complex_str)
                                csi.append(complex_num)
                            self.csi.append(csi)
                            # print(self.csi)
                            limit_counter += 1
                            jump_counter = 0
                            skip_close_prev_time = skip_close_curr_time
        self.t_num = len(self.time)
        self.f_num = len(self.freq)
        t_start = self.time[0]
        self.time = [t - t_start for t in self.time]                            # used to relate first time data points to 0
        self.last_time = float(self.time[-1])
        #print(self.time)
        print("done.")


    def prep(self):
        """ Converts and arranged read data into arrays for computation. """
        print("Starting to prep data for '" + self.name + "'... ", end='')
        # Create input vector X of two dimensions, time and frequency.
        f = np.asarray(self.freq, dtype="float64")
        t = np.asarray(self.time, dtype="float64")
        # print(self.freq)
        # print(self.time)
        f_mesh, t_mesh = np.meshgrid(f, t)
        self.x = np.dstack([f_mesh, t_mesh]).reshape(-1,2)
        # print(self.x)

        # Create output vector Y of two dimensions, real and imaginary part.
        y_complex = np.asarray(self.csi)                                        # convert to numpy arrays
        self.y_complex = y_complex.reshape(-1,1)
        a = y_complex.real                                                      # get real parts into new array 'a'
        b = y_complex.imag                                                      # get imaginary parts into new array 'b'
        # print("a", a)
        # print("b", b)
        self.y = np.dstack([a, b]).reshape(-1,2)                                # construct output vector Y of two dimensions
        # print("self.y", self.y)
        # Create output vectors with absolute value and angle respectively.
        self.y_abs = np.absolute(y_complex).reshape(-1,1)                       # construct output vector 'Y_abs' with absolute value for each data point
        self.y_ang = np.angle(y_complex).reshape(-1,1)                          # construct output vector 'Y_ang' with angle for each data point
        self.y_abs_ang = np.dstack([self.y_abs, self.y_ang]).reshape(-1,2)
        # print(self.y_abs_ang)
        print("done.")

    def normalize(self, vec): # normalize to (-1,1)
        vec = np.asarray(vec)
        amin = np.amin(vec)
        self.norms.append(amin)
        vec = vec - amin
        amax = np.amax(vec)
        self.norms.append(amax)
        vec = 2 * vec/amax - 1
        return vec

    def norm(self):
        """ Normalizes input data X, so all data is within (-1, 1). """
        print("Starting to normalize data for '" + self.name + "'... ", end='')
        # Normalize input
        f = self.x[:,0]
        t = self.x[:,1]
        f = self.normalize(f).reshape(-1,1)
        t = self.normalize(t).reshape(-1,1)
        self.x = np.dstack([f, t]).reshape(-1,2)
        self.time = self.normalize(self.time)
        self.freq = self.normalize(self.freq)
        # Normalize output
        # a = self.y[:,0]
        # b = self.y[:,1]
        # a = self.normalize(a).reshape(-1,1)
        # b = self.normalize(b).reshape(-1,1)
        # self.y = np.dstack([a, b]).reshape(-1,2)
        #
        # abs = self.y_abs
        # ang = self.y_ang
        # abs = self.normalize(abs).reshape(-1,1)
        # ang = self.normalize(ang).reshape(-1,1)
        # self.y_abs_ang = np.dstack([abs, ang]).reshape(-1,2)
        # self.y_abs = abs
        # self.y_ang = ang
        print("done.")

    def unnormalize(self, vec, amin, amax): # Unnormalize agian
        print(amin, amax)
        vec = np.asarray(vec)
        vec = (amax * vec + 1)/2
        vec = vec + amin
        return vec

    def unnorm(self):
        """ Normalizes input data X, so all data is within (-1, 1). """
        print("Starting to normalize data for '" + self.name + "'... ", end='')
        # Normalize input
        f = self.x[:,0]
        t = self.x[:,1]
        f = self.unnormalize(f, self.norms[0], self.norms[1]).reshape(-1,1)
        t = self.unnormalize(t, self.norms[2], self.norms[3]).reshape(-1,1)
        self.x = np.dstack([f, t]).reshape(-1,2)
        self.time = self.unnormalize(self.time, self.norms[4], self.norms[5])
        self.freq = self.unnormalize(self.freq, self.norms[6], self.norms[7])
        # Normalize output
        a = self.y[:,0]
        b = self.y[:,1]
        a = self.unnormalize(a, self.norms[8], self.norms[9]).reshape(-1,1)
        b = self.unnormalize(b, self.norms[10], self.norms[11]).reshape(-1,1)
        self.y = np.dstack([a, b]).reshape(-1,2)

        abs = self.y_abs
        ang = self.y_ang
        abs = self.unnormalize(abs, self.norms[12], self.norms[13]).reshape(-1,1)
        ang = self.unnormalize(ang, self.norms[14], self.norms[15]).reshape(-1,1)
        self.y_abs_ang = np.dstack([abs, ang]).reshape(-1,2)
        self.y_abs = abs
        self.y_ang = ang
        print("done.")


    def datetime_to_sec(self, datetime_str):                                    # datetime_ = '2019-03-31 14:56:27.198'
        """ Converts string of date and time to seconds. """
        time_str = datetime_str.split(" ")[1]                                   # time_str = '14:56:27.198'
        hms = time_str.split(".")[0]                                            # hms = '14:56:27'
        ms = float(time_str.split(".")[1])/1000                                 # ms = '0.198'
        time_ = datetime.strptime(hms, '%H:%M:%S') - datetime(1900,1,1)         # removes "standard date" 1900-01-01
        s = float(time_.total_seconds())
        return s + ms                                                           # return time in seconds


    def norm_effect(self):
        abs_mesh = self.y_abs.reshape(self.t_num, self.f_num)
        # print(abs_mesh)
        # Calculate mean effect for transfer function sampled over time
        sum = 0
        for t in range(self.t_num):
            sum += np.sum(abs_mesh[t])
        abs_mean = sum/self.t_num
        # print("abs_mean:", abs_mean)
        # Normalize effect for transfer function for each sample over time.
        abs_norm = np.zeros(shape=(self.t_num, 52))
        for t in range(self.t_num):
            curr_sum = np.sum(abs_mesh[t])
            norm_ratio = abs_mean/curr_sum
            # print("norm_ratio:", norm_ratio)
            curr = abs_mesh[t]
            curr_norm = norm_ratio*abs_mesh[t]
            # print("curr:", curr)
            # print("curr_norm:", curr_norm)
            # print("shape:", curr.shape)
            # print(norm_ratio*np.sum(abs_mesh[t]))
            abs_norm[t] = curr_norm
        # print("abs_norm:", abs_norm)
        self.y_abs = abs_norm.reshape(-1,1)
        self.y_abs_ang = np.dstack([self.y_abs, self.y_ang]).reshape(-1,2)
        a = np.multiply(self.y_abs, np.cos(self.y_ang))
        b = np.multiply(self.y_abs, np.sin(self.y_ang))
        self.y = np.dstack([a, b]).reshape(-1,2)
        self.y_complex = a + (1j)*b

    def norm_phase(self):
        ang = self.y_ang.reshape(self.t_num, self.f_num)
        c = self.y_complex.reshape(self.t_num, self.f_num)
        res = []
        for time in range(self.t_num):
            phi1 = ang[time][0]
            for freq in range(self.f_num):
                res.append(c[time][freq] * np.exp((-1j)*phi1))
        res = np.asarray(res)
        self.y_complex = res.reshape(-1,1)
        self.y_abs = np.absolute(res).reshape(-1,1)
        self.y_ang = np.angle(res).reshape(-1,1)
        self.y_abs_ang = np.dstack([self.y_abs, self.y_ang]).reshape(-1,2)
        a = np.multiply(self.y_abs, np.cos(self.y_ang))
        b = np.multiply(self.y_abs, np.sin(self.y_ang))
        self.y = np.dstack([a, b]).reshape(-1,2)






    def setup(self, x_dims, x_start, y_dims, y_start, jump):
        if x_dims==2:
            x_train = torch.from_numpy(self.x[x_start::jump]).type('torch.FloatTensor').contiguous()
            x_test = torch.from_numpy(self.x[x_start+jump/2::jump]).type('torch.FloatTensor').contiguous()
        if y_dims==1:
            y_train_a = torch.from_numpy(self.y[:,0][x_start::jump]).type('torch.FloatTensor').contiguous()
            y_test_a = torch.from_numpy(self.y[:,0][x_start+jump/2::jump]).type('torch.FloatTensor').contiguous()
            y_train_b = torch.from_numpy(self.y[:,0][x_start::jump]).type('torch.FloatTensor').contiguous()
            y_test_b = torch.from_numpy(self.y[:,0][x_start+jump/2::jump]).type('torch.FloatTensor').contiguous()
            return x_train, x_test, y_train_a, y_test_a, y_train_b, y_test_b
        elif y_dims==2:
            y_train = torch.from_numpy(self.y[x_start::x_jump]).type('torch.FloatTensor').contiguous()
            y_test = torch.from_numpy(self.y[x_start+jump/2::jump]).type('torch.FloatTensor').contiguous()
            return x_train, x_test, y_train, y_test


    def get_slice(self, xy, vector, time=None, freq=None):
        if xy=='x':
            col1 = vector[:,0]
            col2 = vector[:,1]
            mesh1 = col1.reshape(self.t_num, self.f_num)
            mesh2 = col2.reshape(self.t_num, self.f_num)
            # print("mesh1:", mesh1)
            # print("mesh2:", mesh2)
            if (time!=None and freq!=None):
                print("Can't take slice for both time and frequency...")
            elif time!=None:
                time_slice1 = mesh1[time,:]
                time_slice2 = mesh2[time,:]
                time_slice = np.dstack([time_slice1, time_slice2]).reshape(-1,2)
                return time_slice
            elif freq!=None:
                freq_slice1 = mesh1[:,freq]
                freq_slice2 = mesh2[:,freq]
                freq_slice = np.dstack([freq_slice1, freq_slice2]).reshape(-1,2)
                return freq_slice
        elif xy=='y':
            if time!=None:
                time_slice = vector[time*self.f_num:(time+1)*self.f_num]
                return time_slice
                pass
            elif freq!=None:
                freq_slice = vector[freq::self.f_num]
                return freq_slice

    def get_mesh(self, Y, y_dim, norm):
        freq = self.freq
        time = self.time
        if norm is True:
            freq = normalize(freq)
            time = normalize(time)
        mesh_f, mesh_t = np.meshgrid(freq, time)
        if y_dim==1:
            mesh_y = Y.reshape(self.f_num, self.t_num)
            return mesh_f, mesh_t, mesh_y
        elif y_dim==2:
            mesh_a = Y[:,0].reshape(self.f_num, self.t_num)
            mesh_b = Y[:,1].reshape(self.f_num, self.t_num)
            return mesh_f, mesh_t, mesh_a, mesh_b

    def split(self, vector, portion_train, portion_test):
        # Initalize vectors of training and data points.
        vector_train = np.array([0,0])
        vector_test = np.array([0,0])
        portion_size = portion_train+portion_test
        portions = len(vector)//(portion_size)
        train_tot = portions*portion_train
        test_tot = portions*portion_test
        for i in range(portions):
            # Stack training points.
            start = i*portion_size
            end = i*portion_size+portion_train
            vector_train = np.vstack([vector_train, vector[start:end:,]])
            # Stack test points.
            start = i*portion_size+portion_train
            end = i*portion_size+portion_train+portion_test
            vector_test = np.vstack([vector_test, vector[start:end:,]])
        # Stack last points that are over as test points.
        vector_test = np.vstack([vector_test, vector[end::,]])
        return vector_train[1:].reshape(-1,2), vector_test[1:].reshape(-1,2)

