#! /usr/bin/env python
#%%
import os
import sys
sys.path.append("./src")
import numpy as np
import pandas as pd
from scipy import interpolate
from pathways import one_path
import joblib
from joblib import Parallel, delayed
import pickle
#%%
θ_list = pd.read_csv("./data/model144.csv", header=None)[0].to_numpy()/1000
Es = pd.read_excel("./data/iamc_db.xlsx", header=None, sheet_name="data", usecols="F:O")[15:].to_numpy()
years = np.array([0, 5, 10, 20, 30, 40, 50, 60, 70, 80])
#e_funcs = [interpolate.interp1d(years, Es[i]) for i in range(Es.shape[0])]
e_funcs = interpolate.interp1d(years, Es)
#%%
t_annual = np.arange(years[-1])
Et_annual = e_funcs(t_annual)
#%%
γ_3_list = np.linspace(0, 1./3, 20)
N = 100_000
args_damage = (0.00017675, 0.0044, γ_3_list)
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
pickle.dump(res, open("res_simul_large", "wb"))