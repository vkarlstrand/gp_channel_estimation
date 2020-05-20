""" dgpr2d1_predict_freq_results.py

    See file dgpr2d1_predict_freq.py for more info.

    RESULTS:
    Predicting with different portions of training data and test data in frequency.
    Set by TRAIN and TEST which dictates fow many TRAIN points should be taken followed
    by TEST points. E.g. if TRAIN = 1 and TEST = 3, then the data vectors will be on the
    form: [train test test test train test test test train ... etc.]

    DGP

    EPOCHS = 1500
    LEARNING_RATE = 0.0075
    SAMPLES = 64
    NUM_INDUCING = 256

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





    GP
    See arrays below.


"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch
import gpytorch


rmse_y1_dgp = [0.04114291071891785,
               0.09019124507904053,
               0.06880462169647217,
               0.07794246822595596,
               0.08086462318897247,
               0.09180738031864166,
               0.11572924256324768,
               0.09954195469617844,
               0.20513486862182617]

rmse_y2_dgp = [0.04577477648854256,
               0.0535380020737648,
               0.05571197345852852,
               0.06084160879254341,
               0.0675506740808487,
               0.21892714500427246,
               0.20939308404922485,
               0.2484954595565796,
               0.3005870282649994]

rmse_y1_gp = [0.03965378552675247,
              0.0543326735496521,
              0.05452754348516464,
              0.08637623488903046,
              0.10944835841655731,
              0.13355526328086853,
              0.1487177163362503,
              0.14796212315559387,
              0.15305669605731964]

rmse_y2_gp = [0.04438502714037895,
              0.053744126111269,
              0.05617540702223778,
              0.06186584383249283,
              0.06679931282997131,
              0.2131744623184204,
              0.21136434376239777,
              0.2010646015405655,
              0.27633875608444214]



# 1/2

rmse_x =  [1, 1/2, 1/3, 1/4, 1/5, 1/7, 1/9, 1/11, 1/13]
rmse_x =  [9, 8, 7, 6, 5, 4, 3, 2, 1]
rmse_x_ticks =  ["$\\frac{1}{1}$", "$\\frac{1}{2}$", "$\\frac{1}{3}$", "$\\frac{1}{4}$",
        "$\\frac{1}{5}$", "$\\frac{1}{7}$", "$\\frac{1}{9}$",
        "$\\frac{1}{11}$", "$\\frac{1}{13}$"]

matplotlib.rcParams.update({'font.size': 16})

fig1 = plt.figure(figsize=(8, 3))
# plt.suptitle("DGPR and GPR - RMSE of real part")
ax1 = fig1.add_subplot()
ax1.plot(rmse_x, rmse_y1_dgp,
                 c='k', linestyle='-', lw=2, label="DGP")
ax1.plot(rmse_x, rmse_y1_gp,
                 c='k', linestyle='--', lw=2, label="GP")


ax1.legend(ncol=2)
# ax1.set_xlim([0, 13])
# ax1.set_ylim([0, 0.5])
ax1.set_xlabel("Proportion of $\mathcal{D}_{TR}$")
ax1.set_ylabel("RMSE")
ax1.set_xticks(rmse_x)
ax1.set_xticklabels(rmse_x_ticks, fontsize=20)
ax1.set_yscale('log')
ax1.grid(which="minor", alpha=0.25)
ax1.grid(which="major", alpha=0.5)
ax1.set_yticks([0, 0.05, 0.1, 0.2, 0.3])
ax1.set_yticklabels(["0", "0.05", "0.1", "0.2", "0.3"])
ax1.set_ylim([0.025, 0.35])
fig1.tight_layout(pad=2)

fig2 = plt.figure(figsize=(8, 3))
# plt.suptitle("DGPR and GPR - RMSE of imginary part")
ax2 = fig2.add_subplot()
ax2.plot(rmse_x, rmse_y2_dgp,
                 c='k', linestyle='-', lw=2, label="DGP")
ax2.plot(rmse_x, rmse_y2_gp,
                 c='k', linestyle='--', lw=2, label="GP")


ax2.legend(ncol=2)
# ax1.set_xlim([0, 13])
# ax1.set_ylim([0, 0.5])
ax2.set_xlabel("Proportion of $\mathcal{D}_{TR}$")
ax2.set_ylabel("RMSE")
ax2.set_xticks(rmse_x)
ax2.set_xticklabels(rmse_x_ticks, fontsize=20)
ax2.set_yscale('log')
ax2.set_yticks([0, 0.05, 0.1, 0.2, 0.3])
ax2.set_yticklabels(["0", "0.05", "0.1", "0.2", "0.3"])
ax2.grid(which="minor", alpha=0.25)
ax2.grid(which="major", alpha=0.5)
ax2.set_ylim([0.025, 0.35])

fig2.tight_layout(pad=2)

fig3 = plt.figure(figsize=(8, 3))
plt.suptitle("DGPR and GPR - RMSE")
ax3 = fig3.add_subplot()
ax3.plot(rmse_x, rmse_y1_dgp,
                 c='k', linestyle='-', lw=1.5, label="DGP - Real")
ax3.plot(rmse_x, rmse_y1_gp,
                 c='k', linestyle='--', lw=1.5, label="GP - Real")
ax3.plot(rmse_x, rmse_y2_dgp,
                 c='k', linestyle='-', lw=1.5, label="DGP- Imaginary")
ax3.plot(rmse_x, rmse_y2_gp,
                 c='k', linestyle='--', lw=1.5, label="GP - Imaginary")


ax3.legend(ncol=2)
# ax1.set_xlim([0, 13])
# ax1.set_ylim([0, 0.5])
ax3.set_xlabel("Proportion of $\mathcal{D}_{TR}$")
ax3.set_ylabel("$\log$ RMSE $\mathcal{D}_{TE}$")
ax3.set_xticks(rmse_x)
ax3.set_xticklabels(rmse_x_ticks, fontsize=20)
ax3.set_yticks([0, 0.1, 0.2, 0.3])
ax3.set_yticklabels(["0", "0.1", "0.2", "0.3"])
ax3.set_yscale('log')
ax3.grid(which="minor", alpha=0.5)

fig3.tight_layout(pad=2)

plt.show()