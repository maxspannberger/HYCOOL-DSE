from math import pi

N_check_valves = 4
N_shutoff_valves = 8
pipe_diameter = 0.015
m_dot = 0.09
rho = 70.8

# full bore valves - inner diameter is same as pipe diameter

W_check_valve = 2440.4 * pipe_diameter ** 2 + 44.903 * pipe_diameter + 1.2973
W_shutoff_valve = 5
W_valves = N_check_valves * W_check_valve + N_shutoff_valves * W_shutoff_valve

Cv_check_valve = 22413 * pipe_diameter ** 2.0817
Cv_shutoff_valve = 1173.6 * pipe_diameter - 10.19

Q = (m_dot / rho) * 15850.3  # volumetric flow rate in GPM (gallons per minute)
S_g = rho / 999  # specific gravity (dimensionless)

pressure_drop_check = (S_g / ((Cv_check_valve / Q) ** 2)) / 14.504  # convert from psi to bar
pressure_drop_shutoff = (S_g / ((Cv_shutoff_valve / Q) ** 2)) / 14.504  # convert from psi to bar

total_pressure_drop = N_check_valves * pressure_drop_check + N_shutoff_valves * pressure_drop_shutoff

print(f"Total weight of valves: {W_valves:.2f} kg")
print(f"Total pressure drop: {total_pressure_drop:.2f} bar")

# PRVs are not sized due to complex sizing procedure, justify with "clogged pipe" calculation

