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
res = pickle.load(open("res_simul_extended", "rb"))
n_sample = len(res)
T, = res[0][0].shape
#%%
# years = np.array([0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90])
# if not os.path.exists("yt_mat_1000.npy"):
yt_mat = np.zeros((n_sample, T))
for i in range(n_sample):
    yt, _ = res[i]
    yt_mat[i] = yt
np.save("data/yt_mat_extended", yt_mat)

yt_mat = np.load("data/yt_mat_extended.npy")
## damage
# if not os.path.exists("data/nt_mat_1000.npy"):
nt_mat = np.zeros((n_sample, T))
for i in range(n_sample):
    _, lognt = res[i]
    nt_mat[i] = np.exp(-lognt)
np.save("data/nt_mat_extended", nt_mat)

nt_mat = np.load("data/nt_mat_extended.npy")

# for i in range(n_sample):
    # plt.plot(nt_mat)
# plt.xlabel("Years (starts from 2015)")
# plt.title(r"Damage, $\exp (-n)$")
# plt.savefig("./figures/damage_all_large")
# plt.show()
#%%
n_9 = np.quantile(nt_mat, 0.9, axis=0)
n_5 = np.mean(nt_mat, axis=0)
n_1 = np.quantile(nt_mat, 0.1, axis=0)
#%%
plt.figure()
plt.plot(n_9, label=".9 quantile")
plt.plot(n_5, label="mean")
plt.plot(n_1, label=".1 quantile")
plt.legend()
plt.xlabel("Years (starts from 2015)")
plt.title(r"Damage, $\exp (-n)$")
plt.ylim(0.45, 1.05)
# plt.savefig("./figures/damage_quantile_large")
# plt.show()
#%%
res_pulsed = pickle.load(open("res_simul_extended_pulsed", "rb"))
n_sample = len(res_pulsed)
T, = res[0][0].shape
#%%
# if not os.path.exists("data/yt_mat_1000_pulsed.npy"):
yt_mat = np.zeros((n_sample, T))
for i in range(n_sample):
    yt, _ = res_pulsed[i]
    yt_mat[i] = yt
np.save("data/yt_mat_extended_pulsed", yt_mat)

yt_mat_pulsed = np.load("data/yt_mat_extended_pulsed.npy")
## damage
# if not os.path.exists("data/nt_mat_1000_pulsed.npy"):
nt_mat = np.zeros((n_sample, T))
for i in range(n_sample):
    _, lognt = res_pulsed[i]
    nt_mat[i] = np.exp(-lognt)
np.save("data/nt_mat_extended_pulsed", nt_mat)

nt_mat_pulsed = np.load("data/nt_mat_extended_pulsed.npy")

# for i in range(n_sample):
#     plt.plot(nt_mat)
# plt.xlabel("Years (starts from 2015)")
# plt.title(r"Damage, $\exp (-n)$")
# # plt.savefig("./figures/damage_all_large")
# plt.show()
#%%
n_9_p = np.quantile(nt_mat_pulsed, 0.9, axis=0)
n_5_p = np.mean(nt_mat_pulsed, axis=0)
n_1_p = np.quantile(nt_mat_pulsed, 0.1, axis=0)
#%%
plt.figure()
plt.plot(n_9_p, label=".9 quantile")
plt.plot(n_5_p, label="mean")
plt.plot(n_1_p, label=".1 quantile")
plt.legend()
plt.xlabel("Years (starts from 2015)")
plt.title(r"Damage, $\exp (-n)$, with a pulse")
plt.ylim(0.45, 1.05)
# plt.savefig("./figures/damage_quantile_large_pulsed")
# plt.show()
#%%
t = np.arange(0, len(n_9))
plt.figure()
plt.plot(np.cumsum(np.exp( - 0.02 *t) *(np.log(n_9) - np.log(n_9_p)) ) * 1000* 85, label=".9 quantile")
plt.plot(np.cumsum(np.exp( - 0.02 *t) *(np.log(n_5) - np.log(n_5_p)) ) * 1000* 85, label="mean")
plt.plot(np.cumsum(np.exp( - 0.02 *t) *(np.log(n_1) - np.log(n_1_p)) )* 1000* 85, label=".1 quantile")
plt.legend()
plt.xlabel("Years (starts from 2015)")
plt.title(r"Log damage difference, $\log N_t$")
# plt.savefig("./figures/damage_diff_quantile_")
plt.show()
#%%
t = np.arange(0, len(n_9))
plt.figure()
plt.plot(np.cumsum(np.exp( - 0.02 *t) *(np.log(n_9) - np.log(n_9_p)) ) * 1000* 85, label=".9 quantile")
plt.plot(np.cumsum(np.exp( - 0.02 *t) *(np.log(n_5) - np.log(n_5_p)) ) * 1000* 85, label="mean")
plt.plot(np.cumsum(np.exp( - 0.02 *t) *(np.log(n_1) - np.log(n_1_p)) )* 1000* 85, label=".1 quantile")
plt.legend()
plt.xlim(0,100 )
plt.xlabel("Years (starts from 2015)")
plt.title(r"Log damage difference, $\log N_t$")
# plt.savefig("./figures/damage_diff_quantile_")
plt.show()
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