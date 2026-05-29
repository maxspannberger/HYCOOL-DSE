"""
Hydrogen fuel-expander (topping) cycle - first-order power model
================================================================
Per-module model for a turboelectric H2 gas turbine.

Cycle path (the order matters):
  State 1  tank            : cold LH2 at tank pressure
  State 2  after HP pump   : pressurised COLD (cheap, liquid is dense)
  State 3  after heating   : high-P gas, heated by cooling loads + exhaust recuperator
  State 4  after expander  : expanded to combustor delivery pressure, shaft work out

Net topping power = turbine work - pump work.
The fuel then injects into the combustor at state 4 (must stay above combustor pressure).

Real-gas properties via CoolProp (Leachman EoS). ParaHydrogen is used because
stored LH2 is almost entirely para. Do NOT use ideal-gas relations here: cold
high-pressure hydrogen is strongly non-ideal.

Requires: pip install CoolProp
"""

from CoolProp.CoolProp import PropsSI

# ----------------------------------------------------------------------
# Parameters - edit these
# ----------------------------------------------------------------------
fluid   = "ParaHydrogen"
mdot    = 0.31/2      # kg/s, fuel flow per 2.85 MW module (~ P_fuel/LHV at ~50% eff)
P1      = 3e5         # Pa, tank pressure
T1      = 20.0        # K,  tank temperature
eta_pump = 0.70       # HP pump isentropic efficiency
eta_turb = 0.85       # expander isentropic efficiency
dp_hx    = 0.05       # fractional pressure drop across the heating heat exchangers
P4       = 30e5       # Pa, combustor delivery pressure (keep ABOVE combustor pressure)

# ----------------------------------------------------------------------
def run(P2, T3):
    h1 = PropsSI("H", "P", P1, "T", T1, fluid)
    s1 = PropsSI("S", "P", P1, "T", T1, fluid)

    h2s = PropsSI("H", "P", P2, "S", s1, fluid)
    h2  = h1 + (h2s - h1) / eta_pump
    w_pump = h2 - h1

    P3 = (1 - dp_hx) * P2
    h3 = PropsSI("H", "P", P3, "T", T3, fluid)
    s3 = PropsSI("S", "P", P3, "T", T3, fluid)
    q_heat = h3 - h2                      

    h4s = PropsSI("H", "P", P4, "S", s3, fluid)
    h4  = h3 - eta_turb * (h3 - h4s)
    T4  = PropsSI("T", "P", P4, "H", h4, fluid)
    w_turb = h3 - h4

    return {
        "w_pump": w_pump, "w_turb": w_turb, "q_heat": q_heat,
        "P_pump": mdot * w_pump, "P_turb": mdot * w_turb,
        "P_net": mdot * (w_turb - w_pump), "T4": T4,
    }

# ----------------------------------------------------------------------
if __name__ == "__main__":
    P2 = 100e5
    print(f"Fluid={fluid}  mdot={mdot} kg/s  pump->{P2/1e5:.0f} bar  "
          f"turbine->{P4/1e5:.0f} bar  eta_t={eta_turb}\n")
    print(f"{'T3 [K]':>7}{'q_heat[kW]':>12}{'P_turb[kW]':>12}"
          f"{'P_net[kW]':>11}{'T4[K]':>8}")
    for T3 in range(300, 1000, 100):
        r = run(P2, T3)
        print(f"{T3:7d}{r['q_heat']*mdot/mdot/1e3:12.0f}"
              f"{r['P_turb']/1e3:12.1f}{r['P_net']/1e3:11.1f}{r['T4']:8.0f}")

    T3 = 600
    print(f"\nPump-pressure sweep at T3 = {T3} K")
    print(f"{'P2[bar]':>8}{'P_net[kW]':>11}{'T4[K]':>8}")
    for P2 in [50e5, 80e5, 100e5, 130e5, 160e5]:
        r = run(P2, T3)
        print(f"{P2/1e5:8.0f}{r['P_net']/1e3:11.1f}{r['T4']:8.0f}")