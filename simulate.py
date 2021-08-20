#! /usr/bin/env python
#%%
import os
import sys
sys.path.append("./src")
import numpy as np
import pandas as pd
from scipy import interpolate
from pathways import one_path, simulation_Yt, aggregated
import joblib
from joblib import Parallel, delayed
import pickle
import matplotlib.pyplot as plt
#%%
θ_list = pd.read_csv("./data/model144.csv", header=None)[0].to_numpy()/1000
Es = pd.read_csv("data/RCP85_EMISSIONS.csv", header=6)
years = np.array([0, 5, 15, 25, 35, 45, 55, 65, 75, 85])
# e_funcs = [interpolate.interp1d(years, Es[i]) for i in range(Es.shape[0])]
# e_funcs = interpolate.interp1d(years, Es)
#%%
E_extended = Es[Es[Es.columns[0]] >= 2020].to_numpy()[:, 1:].T
E_extended.shape
#%%
if  not os.path.exists("data/Et_annual.npy"):
    t_annual = np.arange(years[-1])
    Et_annual = e_funcs(t_annual)
    np.save("data/Et_annual", Et_annual)
Et_annual = np.load("data/Et_annual.npy")
if  not os.path.exists("data/Et_pulse.npy"):
    pulse = np.zeros(Et_annual.shape)
    pulse[:, 0] = 1
    Et_pulsed = Et_annual + pulse
    np.save("data/Et_pulse", Et_pulsed)
Et_pulsed = np.load("data/Et_pulse.npy")
#%%
pulse = np.zeros(E_extended.shape)
pulse[:,0] = 1
E_extended_pulsed = E_extended + pulse

Et_annual = E_extended
Et_pulsed = E_extended_pulsed
#%%
Et_annual.shape
#%%
γ_3_list = np.linspace(0, 1./3, 20)
N = 1000
args_damage = (0.00017675, 0.0044, γ_3_list)
force_run = False
if not os.path.exists("res_simul_extended") or force_run:
    args_list = []
    for i in range(len(Et_annual)):
        Et = Et_annual[i]
        for j in range(len(θ_list)):
            θ_j = θ_list[j]
            args_list.append([Et, θ_j])

    print("Start simulation")
    number_of_cpu = joblib.cpu_count()
    delayed_funcs = [delayed(one_path)(Et, θ_j, args_damage) for Et, θ_j in args_list for i in range(N)]
    parallel_pool = Parallel(n_jobs=number_of_cpu)
    res = parallel_pool(delayed_funcs)
    # res = simulation_random(args_list) 
    pickle.dump(res, open("res_simul_extended", "wb"))
else:
    print("Results already computed.")

# pulsed
if not os.path.exists("res_simul_extended_pulsed"):
    args_list = []
    for i in range(len(Et_pulsed)):
        Et = Et_pulsed[i]
        for j in range(len(θ_list)):
            θ_j = θ_list[j]
            args_list.append([Et, θ_j])
    print("Start simulation, pulsed")
    number_of_cpu = joblib.cpu_count()
    delayed_funcs = [delayed(one_path)(Et, θ_j, args_damage) for Et, θ_j in args_list for i in range(N)]
    parallel_pool = Parallel(n_jobs=number_of_cpu)
    res = parallel_pool(delayed_funcs)
    # res = simulation_random(args_list) 
    pickle.dump(res, open("res_simul_extended_pulsed", "wb"))
else:
    print("Results already computed.")
#%%
# simulate with mean θ
Yt_list = []
for i in range(Et_annual.shape[0]):
    Yt = simulation_Yt(Et_annual[i], np.mean(θ_list))
    Yt_list.append(Yt)
#%%
θ_mean = np.mean(θ_list)
ϛ = 1.2 * θ_mean
η = 0.032
δ = 0.01
γ_1 = 0.00017675
γ_2 = 2*0.0022
ME_list = []
for i in range(len(Et_annual)):
    # marginal utility of emission
    T = min(len(Yt_list[i]), len(Et_annual[i]) )
    MU_e = η / Et_annual[i, :T]
    # damage component
    MU_e_dmg = (1 - η) / δ * ( (γ_1 + γ_2 * Yt_list[i][:T] ) * θ_mean  + γ_2 * ϛ**2 * Et_annual[i, :T])
    res = {
        "MU_e": MU_e,
        "MU_e_dmg": MU_e_dmg
    }
    ME_list.append(res)
#%%
ME_list[4]["MU_e"].shape
#%%
Yt_list[0]
#%%
# save res
if not os.path.exists("data/res_MU"):
    with open("data/res_MU", "wb") as handle:
        pickle.dump(ME_list, handle)
else:
    print("Results already stored.")
ME_list = pickle.load(open("data/res_MU_long", "rb"))
#%%
plt.plot(Et_annual[4])
#%%
plt.plot(Yt_list[4])
#%%
# discounted sum
ME_sum = []
ME_dmg = []
ME_temp = []
for i in range(len(Et_annual)):
    MU_e_integral = aggregated(ME_list[i]["MU_e"])
    MU_e_dmg_integral = aggregated(ME_list[i]["MU_e_dmg"])
    ME_sum.append(MU_e_integral)
    ME_dmg.append(MU_e_dmg_integral)
    ME_temp.append(MU_e_integral - MU_e_dmg_integral)
#%%
plt.plot(Et_annual[0])
#%%
i = 0
plt.plot(ME_sum[i]*1000, label="aggregated marginal utility of emission")
plt.plot(ME_dmg[i]*1000, label="damage component")
plt.plot(ME_temp[i]*1000, label="temperature anomaly component")
plt.legend()
plt.xlabel("Years")