from math import pi
import numpy as np
import matplotlib.pyplot as plt
# Pipe sizing and insulation

# Assumed values
rho = 71 # hydrogen density
f = 0.013 # friction factor for turbulent flow in smooth pipe
K = 0.02 # minor loss coefficient for fittings and bends
T_LH2 = 20 # temperature of liquid hydrogen
T_ambient = 295 # ambient temperature
h = 10 # convective heat transfer coefficient
k = 0.0173 # thermal conductivity of foam insulation (polyurethane foam)
h_lat = 446000 # latent heat of vaporization for hydrogen [J/g]

# Chosen values
m_dot = 0.071 # mass flow rate
L = 32 # length of the pipe
b = 10 # number of fittings and bends

# Values to trade-off
D = np.linspace(0.006, 0.02, 100) # diameter of the pipe
t = np.linspace(0.05, 0.1, 100) # insulation thickness

K_t = K * b
p_diff = 8 / (pi ** 2 * rho) * ((m_dot ** 2) / (D ** 4))*(f*L/D + K_t) # pressure drop calculation (Re is within f, but the flow is turbulent so this should be deconstructed later)

D = D * 1000 # convert diameter to mm for plotting
p_diff = p_diff / 1000 # convert pressure drop to kPa for plotting

plt.plot(D, p_diff)
plt.xlabel('Diameter (mm)')
plt.ylabel('Pressure Drop (kPa)')
plt.title('Pressure Drop vs Diameter')
plt.grid(True)


D = 0.00752


r1 = (D / 2) + 0.0005
r2 = r1 + t

R_cond = np.log(r2 / r1) / (2 * pi * k)

q_dot = (T_ambient - T_LH2) / R_cond

# 3 mm per meter contraction stainless steel
# foam insulation taken at sea level - conservative

plt.figure(figsize=(8, 5))
plt.plot(t * 1000, q_dot / (h_lat * m_dot / 100)) # convert to boil off percentageloss per meter

plt.xlabel("Insulation thickness [mm]")
plt.ylabel("Heat input per length [W/m]")
plt.title("Heat Input vs Foam Insulation Thickness for D = 7.52 mm")
plt.grid(True)

Q_total = q_dot * L  # total heat input [W]

plt.figure(figsize=(8, 5))
plt.plot(t * 1000, Q_total / (h_lat * m_dot / 100))  # convert to boil off percentage loss per meter

plt.xlabel("Insulation thickness [mm]")
plt.ylabel("Total heat input [W]")
plt.title("Total Heat Input vs Foam Insulation Thickness for D = 7.52 mm")
plt.grid(True)

plt.show()

t = 0.0831

# diameter sized for pressure drop of 1 bar and insulation sized for 1% boil off loss over a 32m lenght pipe


# Densities [kg/m^3]
rho_ss = 8000       # stainless steel
rho_foam = 40       # polyurethane foam
rho_al = 2700       # aluminium

# Geometry [m]
r_i_ss = D / 2     # stainless steel inner radius
t_ss = 0.0005       # stainless steel wall thickness
t_foam = t      # foam insulation thickness
t_al = 0.0005       # aluminium jacket thickness

# Radii [m]
r_o_ss = r_i_ss + t_ss

r_i_foam = r_o_ss
r_o_foam = r_i_foam + t_foam

r_i_al = r_o_foam
r_o_al = r_i_al + t_al

# Mass per meter function
def mass_per_meter(rho, r_i, r_o):
    return rho * pi * (r_o**2 - r_i**2)

# Mass per meter calculations [kg/m]
m_ss = mass_per_meter(rho_ss, r_i_ss, r_o_ss)
m_foam = mass_per_meter(rho_foam, r_i_foam, r_o_foam)
m_al = mass_per_meter(rho_al, r_i_al, r_o_al)

m_total = m_ss + m_foam + m_al

# Print results
print(f"Stainless steel mass per meter: {m_ss:.3f} kg/m")
print(f"Foam insulation mass per meter: {m_foam:.3f} kg/m")
print(f"Aluminium jacket mass per meter: {m_al:.3f} kg/m")
print(f"Total piping mass per meter: {m_total:.3f} kg/m")

print("\nRadii:")
print(f"SS inner radius: {r_i_ss*1000:.2f} mm")
print(f"SS outer radius: {r_o_ss*1000:.2f} mm")
print(f"Foam outer radius: {r_o_foam*1000:.2f} mm")
print(f"Aluminium outer radius: {r_o_al*1000:.2f} mm")