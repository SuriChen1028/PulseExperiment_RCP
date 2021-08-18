import os
import sys
sys.path.append("../src")
import numpy as np

def Γ(y, γ_3, γ_1=0.00017675, γ_2=2*0.0022, y_overline=2.):
    logN = γ_1 * y + γ_2/2 * y**2 + γ_3/2 * (y - y_overline)**2 * (y > y_overline)
    return logN

def Intensity(y, r1=1.5, r2=2.5, y_underline=1.5):
    return r1 * (np.exp(r2 / 2 *
                       (y - y_underline)**2) - 1) * (y >= y_underline)


def one_path(Et, θ, args=(), dt=1, Y0=1.1):
    γ_1, γ_2, γ_3_list = args
    T = len(Et)
    Yt = np.zeros(T+1)
    logNt = np.zeros(T+1)
    Yt[0] = Y0
    logNt[0] = Γ(Yt[0], 0)
    jumped = False
    γ_3_j = 0
    for i in range(T):
        J_y = Intensity(Yt[i])
        jump_prob = J_y * dt
        jump_prob = jump_prob * (jump_prob <= 1) + (jump_prob > 1)
        if jump_prob > 0 and jumped == False:
            # could jump:
            jump_bool = np.random.choice([False,True], size=1, p=[1 - jump_prob, jump_prob])
            if jump_bool:
                # jump occurs
                # to one of the damage functions
                j = np.random.randint(len(γ_3_list))
                γ_3_j = γ_3_list[j]
                logNt[i] = Γ(Yt[i], γ_3_j)
                Yt[i+1] = 2
                jumped = True
            else:
                logNt[i] = Γ(Yt[i], γ_3_j)
                Yt[i+1] = Yt[i] + θ * Et[i] *dt
        else:
            logNt[i] = Γ(Yt[i], γ_3_j)
            Yt[i+1] = Yt[i] + θ * Et[i] *dt
    logNt[i+1] = Γ(Yt[i+1], γ_3_j)
    return Yt, logNt

    
