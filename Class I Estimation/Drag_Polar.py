import numpy as np
import matplotlib.pyplot as plt

Cd0 = 0.02 # -
e = 0.75 # -
W = 31000 # kg
rho = 1.225 # kg/m^3
S = 64 # m^2
b = 28.4 # m
A = b ** 2 / S  # -
V_lst = np.linspace(20, 180,100)

def lift_coeff(W, rho, V, S):
    Cl = W * 9.80665 / (0.5 * rho * V ** 2 * S)
    return Cl

def induced_drag(Cl, e, A):
    Cdinduced = Cl ** 2 / (np.pi * e * A)
    return Cdinduced

def plot_dat_thang(W, rho, S, Cd0, e, A, V_lst):
    Cd_lst = []
    Cl_lst = []
    D_lst = []
    for V in V_lst:
        V = V * 0.5144444
        Cl = lift_coeff(W, rho, V, S)
        Cdi = induced_drag(Cl, e, A)
        Cd = Cd0 + Cdi
        D = Cd * 0.5 * rho * V ** 2 * S
        Cl_lst.append(Cl)
        Cd_lst.append(Cd)
        D_lst.append(D)
    plt.figure()
    plt.plot(V_lst, D_lst)
    plt.figure()
    plt.plot(Cd_lst, Cl_lst)
    plt.show()

plot_dat_thang(W, rho, S, Cd0, e, A, V_lst)