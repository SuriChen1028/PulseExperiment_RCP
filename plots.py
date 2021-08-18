#! /usr/bin/env python
#%%
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
res = pickle.load(open("res_simul", "rb"))
n_sample = len(res)
T, = res[0][0].shape
#%%
# years = np.array([0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90])
yt_mat = np.zeros((n_sample, T))
nt_mat = np.zeros((n_sample, T))
plt.figure()
for i in range(n_sample):
    yt, lognt = res[i]
    yt_mat[i] = yt
    nt_mat[i] = np.exp(-lognt)
    plt.plot(np.exp(-lognt))
plt.xlabel("Years (starts from 2015)")
plt.title(r"Damage, $\exp (-n)$")
plt.savefig("./figures/damage_all")
# plt.show()
#%%
np.save("yt_mat", yt_mat)
np.save("nt_mat", nt_mat)
#%%
plt.figure()
n_9 = np.quantile(nt_mat, 0.9, axis=0)
n_5 = np.mean(nt_mat, axis=0)
n_1 = np.quantile(nt_mat, 0.1, axis=0)
plt.plot(n_9, label=".9 quantile")
plt.plot(n_5, label="mean")
plt.plot(n_1, label=".1 quantile")
plt.legend()
plt.xlabel("Years (starts from 2015)")
plt.title(r"Damage, $\exp (-n)$")
plt.savefig("./figures/damage_quantile")
# plt.show()