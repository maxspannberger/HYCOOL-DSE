from math import pi, cbrt

N_check_valves = 4
N_shutoff_valves = 8
pipe_diameter = 0.012
# m_dot = 0.09
# rho = 70.8

# full bore valves - inner diameter is same as pipe diameter

W_check_valve = 2440.4 * pipe_diameter ** 2 + 44.903 * pipe_diameter + 1.2973
W_shutoff_valve = 5
W_valves = N_check_valves * W_check_valve + N_shutoff_valves * W_shutoff_valve

# Cv_check_valve = 22413 * pipe_diameter ** 2.0817
# Cv_shutoff_valve = 1173.6 * pipe_diameter - 10.19

# Q = (m_dot / rho) * 15850.3  # volumetric flow rate in GPM (gallons per minute)
# S_g = rho / 999  # specific gravity (dimensionless)

# pressure_drop_check = (S_g / ((Cv_check_valve / Q) ** 2)) / 14.504  # convert from psi to bar
# pressure_drop_shutoff = (S_g / ((Cv_shutoff_valve / Q) ** 2)) / 14.504  # convert from psi to bar

# total_pressure_drop = N_check_valves * pressure_drop_check + N_shutoff_valves * pressure_drop_shutoff

print(f"Total weight of valves: {W_valves:.2f} kg")
# print(f"Total pressure drop: {total_pressure_drop:.2f} bar")

# PRVs are not sized due to complex sizing procedure, justify with "clogged pipe" calculation

# PIPE STUFF

pressure = 2.8 * 10 ** 6
vacuum_thickness = 0.02218  # m

t_inner = (pressure * pipe_diameter) / (2 * (170000000 / 3))

if t_inner < 0.00051:
    t_inner = 0.00051
else:
    t_inner = t_inner

t_outer = cbrt(1 / (3 * 418000)) * (pipe_diameter + 2 * vacuum_thickness)

if t_outer < 0.00051:
    t_outer = 0.00051
else:
    t_outer = t_outer

print(f"Inner thickness: {t_inner * 1000:.2f}[mm]")
print(f"Outer thickness: {t_outer * 1000:.2f}[mm]")

rho_stainless_steel = 8000  # kg/m^3
rho_stainless_steel_20_K = 8080  # kg/m^3
rho_MLI = 21  # kg/m^3, estimated value for MLI insulation
vacuum_boundary_thickness = 0.002  # m
pipe_length = 142  # m

V_pipe_inner = pi * (((pipe_diameter / 2) + t_inner) ** 2 - (pipe_diameter / 2) **2)
mass_pipe_inner = V_pipe_inner * rho_stainless_steel_20_K

V_pipe_outer = pi * (((pipe_diameter / 2) + t_inner + vacuum_thickness + t_outer) ** 2 - ((pipe_diameter / 2) + t_inner + vacuum_thickness) **2)
mass_pipe_outer = V_pipe_outer * rho_stainless_steel_20_K

V_MLI = pi * (((pipe_diameter / 2) + t_inner + vacuum_thickness - vacuum_boundary_thickness) ** 2 - ((pipe_diameter / 2) + t_inner + vacuum_boundary_thickness) **2)
mass_MLI = V_MLI * rho_MLI

mass_pipe_per_meter = mass_pipe_inner + mass_pipe_outer + mass_MLI
total_mass_pipe = mass_pipe_per_meter * pipe_length
print(f"Weight of pipe per meter: {mass_pipe_per_meter:.2f} kg")
print(f"Total weight of pipe: {total_mass_pipe:.2f} kg")

# PUMPS

pump_weight = 20
pump_number = 2
pump_total_weight = pump_weight * pump_number
print(f"Total weight of pumps: {pump_total_weight:.2f} kg")

# FITTINS AND BELLOWS



total_system_weight = W_valves + total_mass_pipe + pump_total_weight
print(f"Total system weight: {total_system_weight:.2f} kg")