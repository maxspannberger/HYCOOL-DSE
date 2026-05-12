import numpy as np
import matplotlib.pyplot as plt

# Values to change
bus_V = 3000
P = 6500000
cable_mm = 1.44 # kg/kA/m
L = 32
N_poles = 2
c_f = 1.3


I = P / bus_V
specific_m = N_poles * I * cable_mm / 1000 # kg
m_bus = specific_m * c_f

m_total = m_bus * L

P_sp = P / m_total

print(m_bus, "kg/m")
print(m_total, "kg")
print(P_sp / 1000, "kW/kg")