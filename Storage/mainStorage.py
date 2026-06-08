"""
Main LH2 tank sizing module.
Integrates geometric design (geomDesign), thermal insulation (thermalDesign),
and structural wall sizing (lh2_tank_trade) into a single TankResult.
"""

from dataclasses import dataclass
from CoolProp.CoolProp import PropsSI

from geomDesign import GeomDesign, Geometry
from thermalDesign import ThermalDesign, InsulationResult
from lh2_tank_trade import t_hoop_stress, material_options

FLUID = 'parahydrogen'
BAR   = 1e5   # Pascals per bar


@dataclass(frozen=True)
class TankResult:
    # ---- Geometry (from geomDesign) ----
    geom:              Geometry

    # ---- Insulation (from thermalDesign) ----
    insulation:        InsulationResult

    # ---- Structural wall ----
    t_wall:            float   # wall thickness [m]
    m_wall:            float   # wall mass [kg]

    # ---- System masses ----
    m_empty:           float   # empty tank mass: wall + insulation [kg]
    m_full:            float   # filled tank mass: m_LH2 + m_empty [kg]

    # ---- Figure of merit ----
    gravimetric_index: float   # m_LH2 / (m_LH2 + m_wall + m_ins) [-]

    # ---- Stored inputs ----
    T_fill:            float   # fill temperature [K]

    def print_summary(self):
        self.geom.print_summary()
        self.insulation.print_summary()
        w = 58
        print("=" * w)
        print(f"{'  WALL & SYSTEM SUMMARY':^{w}}")
        print("=" * w)
        print(f"  {'Fill temperature':<30}  {self.T_fill:>10.2f}  K")
        print(f"  {'Wall thickness':<30}  {self.t_wall * 1000:>10.3f}  mm")
        print(f"  {'Wall mass':<30}  {self.m_wall:>10.3f}  kg")
        print(f"  {'Insulation mass':<30}  {self.insulation.m_ins:>10.3f}  kg")
        print("-" * w)
        print(f"  {'Empty tank mass':<30}  {self.m_empty:>10.3f}  kg")
        print(f"  {'Full tank mass':<30}  {self.m_full:>10.3f}  kg")
        print(f"  {'Gravimetric index':<30}  {self.gravimetric_index:>10.4f}  -")
        print("=" * w + "\n")


def sizeTank(
    m_LH2:           float,
    p_fill:          float,
    p_vent:          float,
    p_ext:           float,
    T_fill:          float,
    tank_material:   str,
    tau_H_hours:     float,
    y_max:    float = 0.97,
    phi:      float = 1.0,
    psi:      float = 1.0,
    Lambda:   float = 0.55,
) -> TankResult:
    """
    Size the complete LH2 storage tank.

    Parameters
    ----------
    m_LH2           : hydrogen mass to store [kg]
    p_fill          : fill pressure [bar]
    p_vent          : vent pressure [bar]
    p_ext           : external (ambient) pressure for wall sizing [Pa]
    T_fill          : fill temperature [K]
    tank_material   : key in material_options, e.g. 'Al-2219-T87'
    tau_H_hours     : maximum no-vent hold time [hours]
    y_max           : liquid volume fraction at which venting begins [-]
    phi             : tank shape ratio a/c [-]
    psi             : tank shape ratio b/c [-]
    Lambda          : shell fraction ls / (ls + 2b) [-]

    Returns
    -------
    TankResult
    """
    if tank_material not in material_options:
        raise ValueError(f"Unknown tank_material '{tank_material}'. "
                         f"Available: {list(material_options)}")

    mat     = material_options[tank_material]
    tau_H_s = tau_H_hours * 3600.0

    # 1. Geometry ----------------------------------------------------------
    gd      = GeomDesign(p_vent=p_vent, p_fill=p_fill, y_max=y_max)
    yl_0    = gd.calculateInitialLiquidMassFraction(yl_vent=y_max)
    # Saturated liquid density at fill pressure for tank volume sizing.
    # T_fill may equal T_sat which puts CoolProp on the phase boundary;
    # Q=0 is the robust way to query saturated liquid properties.
    rho_lh2 = PropsSI('D', 'P', p_fill * BAR, 'T', T_fill, FLUID)
    V       = gd.calculateTankVolume(rho_H2=rho_lh2, m_H2=m_LH2, yl_0=yl_0)
    geom    = gd.calculateTankGeometry(V, phi=phi, psi=psi, Lambda=Lambda)

    # 2. Maximum allowable heat leak ---------------------------------------
    td = ThermalDesign()
    Q_leak, Ei, Ef = td.calculateMaxHeatLeak(
        V, yl_0, p_fill, p_vent, y_max, tau_H_s
    )

    # 3. Insulation sizing -------------------------------------------------
    insulation = td.sizeVacuumInsulation(geom, Q_leak, p_fill_bar=p_fill)

    # 4. Wall thickness (hoop stress)
    t_wall = t_hoop_stress(
        P_int     = p_vent * BAR,
        P_ext     = p_ext,
        D         = 2.0 * geom.a,
        S_t       = mat['S_t'],
        spherical = False,
    )

    # 5. Masses ------------------------------------------------------------
    m_wall  = geom.A_tank * t_wall * mat['density']
    m_empty = m_wall + insulation.m_ins
    m_full  = m_LH2 + m_empty

    # 6. Gravimetric index -------------------------------------------------
    gravimetric_index = m_LH2 / (m_LH2 + 2 * m_wall + insulation.m_ins)

    return TankResult(
        geom              = geom,
        insulation        = insulation,
        t_wall            = t_wall,
        m_wall            = m_wall,
        m_empty           = m_empty,
        m_full            = m_full,
        gravimetric_index = gravimetric_index,
        T_fill            = T_fill,
    )


if __name__ == "__main__":
    # ------------------------------------------------------------------ #
    #  USER INPUTS                                                         #
    # ------------------------------------------------------------------ #
    m_LH2           = 500         # hydrogen mass [kg]
    p_fill          = 1.0            # fill pressure [bar]
    p_vent          = 1.5           # vent pressure [bar]
    p_ext           = 37600.0        # external pressure for wall sizing [Pa]  (e.g. 101325 = SL, 37600 = FL250)
    T_fill          = 20          # fill temperature [K]
    tank_material   = 'Al-2219-T87'  # see material_options in lh2_tank_trade.py
    tau_H_hours     = 24.0           # no-vent holding time [hours]
    phi             = 1.0            # tank shape: a/c (major/minor radius ratio) [-]
    psi             = 1.0            # tank shape: b/c (cap height ratio) [-]
    Lambda          = 0.55           # tank shape: ls/(ls + 2b) (shell fraction) [-]
    # ------------------------------------------------------------------ #

    w = 58
    print("\n" + "=" * w)
    print(f"{'  TANK SIZING INPUTS':^{w}}")
    print("=" * w)
    print(f"  {'LH2 mass':<30}  {m_LH2:>10.1f}  kg")
    print(f"  {'Fill pressure':<30}  {p_fill:>10.2f}  bar")
    print(f"  {'Vent pressure':<30}  {p_vent:>10.2f}  bar")
    print(f"  {'External pressure':<30}  {p_ext:>10.0f}  Pa")
    print(f"  {'Fill temperature':<30}  {T_fill:>10.2f}  K")
    print(f"  {'Tank material':<30}  {tank_material:>10}")
    print(f"  {'Holding time':<30}  {tau_H_hours:>10.1f}  h")
    print(f"  {'phi  (a/c)':<30}  {phi:>10.2f}  -")
    print(f"  {'psi  (b/c)':<30}  {psi:>10.2f}  -")
    print(f"  {'Lambda (ls/(ls+2b))':<30}  {Lambda:>10.2f}  -")
    print("=" * w)

    result = sizeTank(
        m_LH2           = m_LH2,
        p_fill          = p_fill,
        p_vent          = p_vent,
        p_ext           = p_ext,
        T_fill          = T_fill,
        tank_material   = tank_material,
        tau_H_hours     = tau_H_hours,
        phi             = phi,
        psi             = psi,
        Lambda          = Lambda,
    )

    result.print_summary()
