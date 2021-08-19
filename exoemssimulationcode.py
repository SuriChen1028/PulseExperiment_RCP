import numpy as np
import pandas as pd
import ray
from scipy import interpolate
import matplotlib.pyplot as plt
import plotly.offline as pyo
import plotly.io as pio
import os
pio.templates.default = "none"
pyo.init_notebook_mode()
γ3_list = np.linspace(0,1/3,20)
θ_list = pd.read_csv("model144.csv", header=None)[0].to_numpy()
θ_list = θ_list/1000
γ1 = 0.00017675
γ2 = 2*0.0022
ray.shutdown()
ray.init(num_cpus=5, num_gpus=1)

@ray.remote
def simulation_jump(γ3_list, θ, iterer, model, y1_0, T, dt, y_lower=1.5, y_upper=2.0):
    Et = np.ones(T + 1)
    y1t = np.zeros(T + 1)
    Damage_func = np.zeros(T + 1)

    df = pd.read_excel("kkk.xlsx",
                       header=None).to_numpy()
    years = [0, 5, 10, 12, 14, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    e_fun = interpolate.interp1d(years, df[model, :])

    r1 = 1.5
    r2 = 2.
    γ3 = 0
    Jump = 0
    for i in range(T + 1):
        if Jump == 0:
            Et[i] = e_fun(i)
            Intensity = r1 * (np.exp(r2 / 2 * (y1_0 - y_lower) ** 2) - 1) * (y1_0 >= y_lower)
            rng = np.random.default_rng(iterer)
            Jump = rng.poisson(Intensity) * dt
            y1t[i] = y1_0
            Damage_func[i] = γ1 + γ2 * y1t[i]
            y1_0 = y1_0 + Et[i] * θ * dt
            else_loop = 0
            K = i
        elif Jump >= 1:
            if else_loop == 0:
                Jump = 1
                K = i
                γ3 = rng.choice(γ3_list)

            Et[i] = e_fun(i)
            y1t[i] = y1_0
            Damage_func[i] = γ1 + γ2 * y1t[i] + γ3 * (y1t[i] - y_upper) * (y1t[i] > y_upper)
            y1_0 = y1_0 + Et[i] * θ * dt
            else_loop = 1
    result = dict(model=model, Et=Et, y1t=y1t, Damages=Damage_func, γ3=γ3, K=K)

    print(iterer)
    return (result)


@ray.remote
def simulation_jump_pulse(γ3_list, θ, iterer, model, y1_0, T, dt, y_lower=1.5, y_upper=2.0):
    Et = np.zeros(T + 1)
    y1t = np.zeros(T + 1)
    Damage_func = np.zeros(T + 1)

    df = pd.read_excel("/Users/samuelzhao/Documents/GitHub/ClimateUncertainty-twostate/data/kkk.xlsx",
                       header=None).to_numpy()
    years = [0, 5, 10, 12, 14, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    e_fun = interpolate.interp1d(years, df[model, :])

    r1 = 1.5
    r2 = 2.
    Jump = 0
    γ3 = 0
    for i in range(T + 1):
        if Jump == 0:
            Intensity = r1 * (np.exp(r2 / 2 * (y1_0 - y_lower) ** 2) - 1) * (y1_0 >= y_lower)
            rng = np.random.default_rng(iterer)
            Jump_prob = rng.poisson(Intensity) * dt
            Jump_prob = Jump_prob * (Jump_prob <= 1) + (Jump_prob > 1)
            Jump = rng.choice([0, 1], size=1, p=[1 - Jump_prob, Jump_prob])
            if i == 0:
                Et[i] = e_fun(i) + .1
            else:
                Et[i] = e_fun(i)
            y1t[i] = y1_0
            Damage_func[i] = γ1 + γ2 * y1t[i]
            y1_0 = y1_0 + Et[i] * θ * dt
            else_loop = 0
            K = i
        elif Jump >= 1:
            if else_loop == 0:
                Jump = 1
                K = i
                γ3 = rng.choice(γ3_list)

            Et[i] = e_fun(i)
            y1t[i] = y1_0
            Damage_func[i] = γ1 + γ2 * y1t[i] + γ3 * (y1t[i] - y_upper) * (y1t[i] > y_upper)
            y1_0 = y1_0 + Et[i] * θ * dt
            else_loop = 1

    result = dict(model=model, Et=Et, y1t=y1t, Damages=Damage_func, γ3=γ3, K=K)

    print(iterer)
    return (result)

iteration_list = {}
iteration = np.linspace(0, 100, 101, dtype=int)
models    = np.linspace(0,8,9, dtype=int)
iteration_list=[simulation_jump.remote(γ3_list, θ_list[j], iteration[k], models[m], 1.1, 100, 1)
                for k in range(len(iteration))
                for j in range(len(θ_list))
                for m in range(len(models))]
iteration_list = ray.get(iteration_list)

iteration_list_pulse = {}
iteration = np.linspace(0, 100, 101, dtype=int)
models    = np.linspace(0,8,9, dtype=int)
iteration_list_pulse=[simulation_jump_pulse.remote(γ3_list, θ_list[j], iteration[k], models[m], 1.1, 100, 1)
                for k in range(len(iteration))
                for j in range(len(θ_list))
                for m in range(len(models))]
iteration_list_pulse = ray.get(iteration_list_pulse)

i=0
for j in range(len(θ_list)):
        for iterer in iteration:
            Et_list       = iteration_list[i]['Et']
            y1t_list      = iteration_list[i]['y1t']
            γ3_list_p     = iteration_list[i]['γ3']
            K_list        = iteration_list[i]['K']
            model_list    = iteration_list[i]['model']
            Damages_list  = iteration_list[i]['Damages']
            newpath = r'../iteration_{:.3f}/γ3_{:.3f}/model_{:.3f}/θ_{:.3f}'.format(iterer,γ3_list_p, 0, θ_list[j])
            if not os.path.exists(newpath):
                os.makedirs(newpath)
            os.chdir(newpath)
            false = y1t_list < 2.0
            np.save("Et_list", Et_list)
            np.save("y1t_list", y1t_list)
            np.save("γ3_list_p", γ3_list_p)
            np.save('K_list',K_list)
            np.save('model_list', model_list)
            np.save('Damages_list', Damages_list)
            plt.plot(np.exp(-Damages_list))
            i=i+1
plt.show()

i = 0
for iterer in iteration:
    for model in np.linspace(0, 8, 9, dtype=int):
        for j in range(len(θ_list)):
            Et_list = iteration_list_pulse[i]['Et']
            y1t_list = iteration_list_pulse[i]['y1t']
            γ3_list_p = iteration_list_pulse[i]['γ3']
            K_list = iteration_list_pulse[i]['K']
            model_list = iteration_list_pulse[i]['model']
            Damages_list = iteration_list_pulse[i]['Damages']
            newpath = r'../iteration_pulse_{:.3f}/γ3_{:.3f}/model_{:.3f}/θ_{:.3f}'.format(
                iterer, γ3_list_p, model, θ_list[j])
            if not os.path.exists(newpath):
                os.makedirs(newpath)
            os.chdir(newpath)
            np.save("Et_list", Et_list)
            np.save("y1t_list", y1t_list)
            np.save("γ3_list_p", γ3_list_p)
            np.save('K_list', K_list)
            np.save('model_list', model_list)
            np.save('Damages_list', Damages_list)

            plt.plot(np.exp(-Damages_list))
            i = i + 1
plt.show()

