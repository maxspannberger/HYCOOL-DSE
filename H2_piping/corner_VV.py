import numpy as np
import matplotlib.pyplot as plt
import CoolProp.CoolProp as CP

# Import YOUR actual system components
from h2_components import Corner, area
from system_config import H2SystemConfig

# =============================================================================
# 1. Experimental Data from Figure 8b (R/r = 5)
# Because R/r is double R/d, this matches your default curv = 2.5
# =============================================================================
Re_exp = np.array([20000, 26000, 34000, 77000, 100000, 121000])
K_exp  = np.array([0.55,  0.34,  0.37,  0.38,  0.22,   0.17])

# =============================================================================
# 2. Setup Baseline Fluid Conditions
# =============================================================================
c = H2SystemConfig()
fluid = c.fluid
d = c.pipe_default_d
A = area(d)

# Establish a realistic supercritical cryogenic state before the bend
p1 = 2800000.0  # 28 bar
T1 = 30.0       # 30 K
h1 = CP.PropsSI('H', 'P', p1, 'T', T1, fluid)
rho1 = CP.PropsSI('D', 'P', p1, 'T', T1, fluid)
mu1 = CP.PropsSI('V', 'P', p1, 'T', T1, fluid)

# =============================================================================
# 3. Run YOUR Native Code
# =============================================================================
K_model = []

for Re in Re_exp:
    # Reverse engineer the mass flow required to hit the experimental Reynolds number
    # Re = (4 * m_dot) / (pi * d * mu) --> m_dot = (Re * pi * d * mu) / 4
    m_dot = (Re * np.pi * d * mu1) / 4.0
    u1 = m_dot / (rho1 * A)
    
    initial_conditions = (T1, p1, h1, rho1, u1)
    
    # Instantiate YOUR actual component with curv=2.5
    test_bend = Corner(curv=2.5, diameter=d, N_bend=1)
    
    # Run your exact solver logic
    res = test_bend.solve_H2_state(states=None, T_amb=c.T_amb, m_dot=m_dot, 
                                   system=None, initial_conditions=initial_conditions)
    
    # Extract the resulting pressure from your solver
    p2 = res['p'][-1]
    
    # Back-calculate the K factor from the pressure drop your code generated
    dp = p1 - p2
    K_calculated = dp / (0.5 * rho1 * u1**2)
    
    K_model.append(K_calculated)

K_model = np.array(K_model)

# =============================================================================
# 4. Print & Plot Results
# =============================================================================
errors = ((K_model - K_exp) / K_exp) * 100

print("="*55)
print(" SIMULATON CODE CORNER V&V (curv = 2.5) ".center(55))
print("="*55)
print(f"{'Reynolds Number':<18} | {'Exp. K':<8} | {'Code K':<8} | {'Error (%)'}")
print("-" * 55)
for r, k_e, k_m, err in zip(Re_exp, K_exp, K_model, errors):
    print(f"{r:<18.0f} | {k_e:<8.2f} | {k_m:<8.3f} | {err:>7.1f}%")
print("="*55)

plt.figure(figsize=(9, 6))
plt.scatter(Re_exp, K_exp, facecolors='none', edgecolors='black', s=80, linewidths=1.5, 
            label='Exp. Data', zorder=3)

plt.plot(Re_exp, K_model, color='tab:blue', marker='x', markersize=8, linestyle='-', linewidth=2, 
         label='Simulation Code Output', zorder=2)

plt.title('Validation of Simulation Corner Class (curv = 2.5)', fontsize=14, fontweight='bold')
plt.xlabel('Reynolds Number (Re)', fontsize=12)
plt.ylabel('Loss Coefficient (K)', fontsize=12)
plt.xlim(0, 140000)
plt.ylim(0.0, 1.2)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=11)
plt.tight_layout()
plt.show()