#! /usr/bin/env python
#%%
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['savefig.edgecolor'] = 'w'
mpl.rcParams['savefig.facecolor'] = 'w'
mpl.rcParams['savefig.bbox'] = 'tight'
mpl.rcParams['legend.frameon'] = False
#%%
res = pickle.load(open("res_simul_extended_steep", "rb"))
n_sample = len(res)
T, = res[0][0].shape
#%%
# years = np.array([0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90])
# if not os.path.exists("yt_mat_1000.npy"):
yt_mat = np.zeros((n_sample, T))
for i in range(n_sample):
    yt, _ = res[i]
    yt_mat[i] = yt
np.save("data/yt_mat_extended_steep", yt_mat)
#%%
yt_mat = np.load("data/yt_mat_extended_steep.npy")
#%%
## damage
# if not os.path.exists("data/nt_mat_1000.npy"):
lognt_mat = np.zeros((n_sample, T))
for i in range(n_sample):
    _, lognt = res[i]
    lognt_mat[i] = lognt
np.save("data/lognt_mat_extended_steep", lognt_mat)
#%%
lognt_mat = np.load("data/lognt_mat_extended_steep.npy")

# for i in range(n_sample):
    # plt.plot(nt_mat)
# plt.xlabel("Years (starts from 2015)")
# plt.title(r"Damage, $\exp (-n)$")
# plt.savefig("./figures/damage_all_large")
# plt.show()
#%%
n_9 = np.quantile(lognt_mat, 0.9, axis=0)
n_5 = np.mean(lognt_mat, axis=0)
n_1 = np.quantile(lognt_mat, 0.1, axis=0)
#%%
plt.figure()
plt.plot(n_9, label=".9 quantile")
plt.plot(n_5, label="mean")
plt.plot(n_1, label=".1 quantile")
plt.legend()
plt.xlabel("Years (starts from 2015)")
plt.title(r"Damage, $\log N_t$")
# plt.ylim(0., 1.05)
# plt.savefig("./figures/damage_quantile_large")
# plt.show()
#%%
res_pulsed = pickle.load(open("res_simul_extended_pulsed_steep", "rb"))
n_sample = len(res_pulsed)
T, = res[0][0].shape
#%%
# if not os.path.exists("data/yt_mat_1000_pulsed.npy"):
yt_mat = np.zeros((n_sample, T))
for i in range(n_sample):
    yt, _ = res_pulsed[i]
    yt_mat[i] = yt
np.save("data/yt_mat_extended_pulsed_steep", yt_mat)
#%%
yt_mat_pulsed = np.load("data/yt_mat_extended_pulsed_steep.npy")
#%%
## damage
# if not os.path.exists("data/nt_mat_1000_pulsed.npy"):
lognt_mat = np.zeros((n_sample, T))
for i in range(n_sample):
    _, lognt = res_pulsed[i]
    lognt_mat[i] = lognt
np.save("data/lognt_mat_extended_pulsed_steep", lognt_mat)
#%%
lognt_mat_pulsed = np.load("data/lognt_mat_extended_pulsed_steep.npy")

# for i in range(n_sample):
#     plt.plot(nt_mat)
# plt.xlabel("Years (starts from 2015)")
# plt.title(r"Damage, $\exp (-n)$")
# # plt.savefig("./figures/damage_all_large")
# plt.show()
#%%
n_9_p = np.quantile(lognt_mat_pulsed, 0.9, axis=0)
n_5_p = np.mean(lognt_mat_pulsed, axis=0)
n_1_p = np.quantile(lognt_mat_pulsed, 0.1, axis=0)
#%%
plt.figure()
plt.plot(n_9_p - n_9, label=".9 quantile")
plt.plot(n_5_p - n_5, label="mean")
plt.plot(n_1_p - n_1, label=".1 quantile")
plt.legend()
plt.xlabel("Years (starts from 2015)")
plt.title(r"Damage, $\exp (-n)$, with a pulse")
# plt.ylim(0., 1.05)
# plt.savefig("./figures/damage_quantile_large_pulsed")
# plt.show()
#%%
t = np.arange(0, len(n_9))
plt.figure()
plt.plot(np.cumsum(np.exp( - 0.02 *t) *(n_9_p - n_9) ) * 1000* 85, label=".9 quantile")
plt.plot(np.cumsum(np.exp( - 0.02 *t) *(n_5_p - n_5) ) * 1000* 85, label="mean")
plt.plot(np.cumsum(np.exp( - 0.02 *t) *(n_1_p - n_1) )* 1000* 85, label=".1 quantile")
plt.legend()
plt.xlabel("Years (starts from 2020)")
plt.title(r"Cumulative sum of log damage difference, discount rate 0.02")
plt.tight_layout()
# plt.savefig("./figures/damage_diff_quantile_extended_500_steep")
# plt.show()
#%%
t = np.arange(0, len(n_9))
plt.figure()
plt.plot(np.cumsum(np.exp( - 0.02 *t) *(n_9_p - n_9) ) * 1000* 85, label=".9 quantile")
plt.plot(np.cumsum(np.exp( - 0.02 *t) *(n_5_p - n_5) ) * 1000* 85, label="mean")
plt.plot(np.cumsum(np.exp( - 0.02 *t) *(n_1_p - n_1) )* 1000* 85, label=".1 quantile")
plt.legend()
plt.xlim(0,100 )
plt.ylim(0, 4000)
plt.xlabel("Years (starts from 2020)")
plt.title(r"Cumulative sum of log damage difference, discount rate 0.02")
# plt.savefig("./figures/damage_diff_quantile_extended_100_steep")
plt.show()
#%%
#%%
t = np.arange(0, len(n_9))
plt.figure()
plt.plot(np.cumsum(np.exp( - 0.02 *t) *(n_9_p - n_9) ) * 1000* 85, label=".9 quantile")
plt.plot(np.cumsum(np.exp( - 0.02 *t) *(n_5_p - n_5) ) * 1000* 85, label="mean")
plt.plot(np.cumsum(np.exp( - 0.02 *t) *(n_1_p - n_1) )* 1000* 85, label=".1 quantile")
plt.legend()
plt.xlim(0,200 )
plt.ylim(0, 10000)
plt.xlabel("Years (starts from 2020)")
plt.title(r"Cumulative sum of log damage difference, discount rate 0.02")
plt.savefig("./figures/damage_diff_quantile_extended_300_steep")
plt.show()
#%%
t.shape
#%%
np.cumsum(np.exp( - 0.02 *t) *(n_5_p - n_5) )[100]* 1000* 85
#%%
plt.figure()
plt.plot(n_5 - n_5_p, label="mean")
plt.legend()
plt.xlabel("Years (starts from 2015)")
plt.title(r"Damage difference, $\exp (-n)$, mean")
# plt.savefig("./figures/damage_diff_mean")
plt.show()
#%%
plt.plot(yt_mat[-1])
#%%
yt_mat[-1]
#%%
plt.plot(nt_mat[-1])
#%%[
plt.plot(nt_mat[-1][:45])
#%%
## Marginal utility of emission
res_MU = pickle.load(open("data/res_MU_long", "rb"))
i = 0
plt.plot(res_MU[i]["MU_e"] * 1000, label="marginal utility of emission")
plt.plot(res_MU[i]["MU_e_dmg"] * 1000, label="damage component")
plt.plot(res_MU[i]["MU_e"] * 1000 - res_MU[i]["MU_e_dmg"] * 1000, label="temperature anomaly component")
plt.xlabel("Years")
plt.legend()
plt.ylim(0)
plt.title("Model: AIM/CGE, Scenario:SSP3-70(baseline)")
# plt.savefig(f"figures/MU_e_{i}")
plt.show()
#%%
## Marginal utility of emission
res_MU = pickle.load(open("data/res_MU", "rb"))
i = 0
plt.plot(res_MU[i]["MU_e"] * 1000, label="marginal utility of emission")
plt.plot(res_MU[i]["MU_e_dmg"] * 1000, label="damage component")
plt.plot(res_MU[i]["MU_e"] * 1000 - res_MU[i]["MU_e_dmg"] * 1000, label="temperature anomaly component")
plt.xlabel("Years")
plt.legend()
plt.ylim(0)
plt.title("Model: AIM/CGE, Scenario:SSP3-70(baseline)")
# plt.savefig(f"figures/MU_e_{i}")
plt.show()
#%%
## Marginal utility of emission
res_MU = pickle.load(open("data/res_MU", "rb"))
i = 4
plt.plot(res_MU[i]["MU_e"] * 1000, label="marginal utility of emission")
plt.plot(res_MU[i]["MU_e_dmg"] * 1000, label="damage component")
plt.plot(res_MU[i]["MU_e"] * 1000 - res_MU[i]["MU_e_dmg"] * 1000, label="temperature anomaly component")
plt.xlabel("Years")
plt.legend()
plt.title("Model: IMAGE, Scenario: SSP1-19")
# plt.savefig(f"figures/MU_e_{i}")
plt.show()