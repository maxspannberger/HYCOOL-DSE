"""
Main LH2 tank sizing module.
Integrates geometric design (geomDesign), thermal insulation (INSULATE),
and structural wall sizing (lh2_tank_trade) into a single TankResult.
"""

from dataclasses import dataclass
from CoolProp.CoolProp import PropsSI

from geomDesign import GeomDesign, Geometry
from INSULATE import mli_thickness, MLIResult
from lh2_tank_trade import t_hoop_stress, material_options

FLUID = 'parahydrogen'
BAR   = 1e5   # Pascals per bar


@dataclass(frozen=True)
class TankResult:
    # ---- Geometry (from geomDesign) ----
    geom:              Geometry

    # ---- Insulation (from INSULATE) ----
    insulation:        MLIResult
    m_ins:             float   # insulation mass [kg]

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

    # ---- Ullage ----
    ullage_fraction:   float   # vapor (ullage) volume fraction at fill [-]

    # ---- MLI ----
    N_layers:          int     # number of MLI reflector layers [-]

    def print_summary(self):
        ins = self.insulation
        self.geom.print_summary()
        w = 58
        print("=" * w)
        print(f"{'  INSULATION SUMMARY (MLI)':^{w}}")
        print("=" * w)
        print(f"  {'MLI layers':<30}  {ins.N_layers:>10}  -")
        print(f"  {'Blanket thickness':<30}  {ins.thickness * 1000:>10.2f}  mm")
        print(f"  {'Heat flux':<30}  {ins.flux:>10.4f}  W/m2")
        print(f"  {'Allowable heat leak':<30}  {ins.Q_target:>10.2f}  W")
        print(f"  {'Actual heat leak':<30}  {ins.Q_leak:>10.2f}  W")
        print(f"  {'Surface temperature':<30}  {ins.T_s:>10.2f}  K")
        print(f"  {'Insulation mass':<30}  {self.m_ins:>10.3f}  kg")
        print("=" * w)
        print(f"{'  WALL & SYSTEM SUMMARY':^{w}}")
        print("=" * w)
        print(f"  {'Fill temperature':<30}  {self.T_fill:>10.2f}  K")
        print(f"  {'Ullage fraction':<30}  {self.ullage_fraction:>10.4f}  -")
        print(f"  {'Wall thickness':<30}  {self.t_wall * 1000:>10.3f}  mm")
        print(f"  {'Wall mass':<30}  {self.m_wall:>10.3f}  kg")
        print(f"  {'Insulation mass':<30}  {self.m_ins:>10.3f}  kg")
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
    Nbar:     float = 24.0,
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
    Nbar            : MLI layer density [layers/cm]

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

    # 2 & 3. Insulation sizing (heat-leak budget + MLI) --------------------
    insulation = mli_thickness(
        V=V, phi=phi, lam=Lambda,
        P_vent=p_vent * BAR,
        y_l0=yl_0,
        tau_H=tau_H_s,
        T_c=T_fill,
        Nbar=Nbar,
    )
    _RHO_LAYER_AREAL = 0.20   # kg/m² per MLI layer
    m_ins = insulation.N_layers * _RHO_LAYER_AREAL * insulation.A

    # 4. Wall thickness (hoop stress)
    t_wall = t_hoop_stress(
        P_int     = p_vent * BAR,
        P_ext     = p_ext,
        D         = 2.0 * geom.a,
        S_y       = mat['S_y'],
        spherical = False,
    )

    # 5. Masses ------------------------------------------------------------
    m_wall  = geom.A_tank * t_wall * mat['density']
    m_empty = m_wall + m_ins
    m_full  = m_LH2 + m_empty

    # 6. Gravimetric index -------------------------------------------------
    gravimetric_index = m_LH2 / (m_LH2 + 2 * m_wall + m_ins)

    return TankResult(
        geom              = geom,
        insulation        = insulation,
        m_ins             = m_ins,
        t_wall            = t_wall,
        m_wall            = m_wall,
        m_empty           = m_empty,
        m_full            = m_full,
        gravimetric_index = gravimetric_index,
        T_fill            = T_fill,
        ullage_fraction   = 1.0 - yl_0,
        N_layers          = insulation.N_layers,
    )


if __name__ == "__main__":
    # ------------------------------------------------------------------ #
    #  USER INPUTS                                                         #
    # ------------------------------------------------------------------ #
    m_LH2           = 600        # hydrogen mass [kg]
    p_fill          = 1.75          # fill pressure [bar]
    p_vent          = 1.5 * p_fill          # vent pressure [bar]
    p_ext           = 37600.0        # external pressure for wall sizing [Pa]  (e.g. 101325 = SL, 37600 = FL250)
    T_fill          = 15          # fill temperature [K]
    tank_material   = 'Al-2219-T87'  # see material_options in lh2_tank_trade.py
    tau_H_hours     = 24.0           # no-vent holding time [hours]
    phi             = 1.0            # tank shape: a/c (major/minor radius ratio) [-]
    psi             = 1.0            # tank shape: b/c (cap height ratio) [-]
    Lambda          = 0.55           # tank shape: ls/(ls + 2b) (shell fraction) [-]
    Nbar            = 30.0           # MLI layer density [layers/cm]
    
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
    print(f"  {'Nbar (MLI layers/cm)':<30}  {Nbar:>10.2f}  layers/cm")
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
        Nbar            = Nbar,
    )

    result.print_summary()
