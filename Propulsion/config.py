"""
config.py
=========
Single source of truth for all input parameters used by the propulsion
sizing chain.

Layout
------
Each module of the chain (cycle / off-design / dimensional / expander)
has its own dataclass. A top-level `Config` bundles them so a single
`Config()` instance can be passed through the pipeline:

    from config import Config
    cfg = Config()
    cycle    = GasTurbineCycle.from_config(cfg).size()
    od_eval  = OffDesignEvaluator.from_config(cycle, cfg)
    dimsize  = DimensionalSizing.from_config(cycle, cfg).size()
    expander = PistonExpander.from_config(cycle, cfg).size()

Editing inputs
--------------
Either edit the defaults below in place, or build a tweaked copy:

    cfg = Config()
    cfg.cycle.TIT = 1600.0
    cfg.offdesign.P_shaft_OEI = 2.85e6

All dataclasses are mutable on purpose so that downstream studies can
override values without subclassing.


POWERS:
    - OEI:      3038    kW
    - Cruise:   2179    kW
    - Climb:    2747    kW
    - Reserve:  
    - Approach: 690     kW
    - Takeoff:  2800    kW


Off-Design Inputs
-----------------
Parameters that you do not want to define yourself can be left as 'None'
Directly modifying parameters such as:
    T_pre_comp
    P_pre_comp
    Q_regen_max
    etc...
allows you to set the pre-determined limits for a specific off-design scenario
i.e.
    Optimal design:
        T_pre_comp  = 145   K
    Off-design:
        T_pre_comp  = 200   K

"""

from dataclasses import dataclass, field
from typing import List, Optional


# =====================================================================
# Cycle / design-point inputs (was TurbineSizing.GasTurbineCycle args)
# =====================================================================
@dataclass
class CycleConfig:
    # Sizing target
    P_target:       float = 2.179e6           # W, net shaft power target

    # Ambient / flight conditions
    P_ambient:      float = 0.38            # bar, ambient static pressure
    M0:             float = 0.7             # cruise Mach

    # Core cycle
    TIT:            float = 1600.0          # K, turbine inlet temperature
    PR_HPC:         float = 7.0             # HPC pressure ratio
    eta_HPC:        float = 0.92
    eta_CC:         float = 0.995
    eta_CC_p:       float = 0.96            # combustor pressure recovery
    eta_HPT:        float = 0.92
    eta_HEX:        float = 0.92
    eta_mech:       float = 0.99
    eta_diff:       float = 0.97
    mdot_boiloff:   float = 0.02

    # Recuperator
    eta_regen:      float = 0.775
    eta_regen_p:    float = 0.95
    USE_REGEN:      bool  = True
    FULL_EXPANSION: bool  = True
    REGEN_FIRST:    bool  = False
    Regen_Fraction: float = 0.775

    # Hydrogen circuit
    P_pre_comp:     float = 25.0            # bar
    T_pre_comp:     float = 145.0           # K
    PH1:            float = 150.0           # bar
    TH2:            float = 800.0           # K (legacy seed; TH2 is now set
                                            #    dynamically to the H2 HEX hot-
                                            #    side inlet temperature)
    eta_compressor: float = 0.85
    eta_H2T:        float = 0.92
    fluid:          str   = "ParaHydrogen"
    LHV_H2:         float = 120e6           # J/kg

    # Solver
    mdot_f_init:    float = 0.155           # kg/s


# =====================================================================
# Off-design inputs (was OffDesignEvaluator args)
# =====================================================================
@dataclass
class OffDesignConfig:
    TIT_limit:   float = 1900.0             # K, material limit
    TIT_min:     float = 550.0              # K, lower solver bracket
    TIT_tol:     float = 1e-4
    max_iter:    int   = 100

    # If None, defaults to the design-point recuperator duty.
    Q_regen_max: Optional[float] = None     # W

    # H2 feed temperature at off-design (cryogenic feed to the H2 compressor).
    # If None, falls back to the design-point T_pre_comp.
    T_pre_comp:  Optional[float] = 180     # K

    # H2 feed pressure at off-design (cryogenic feed to the H2 compressor).
    # If None, falls back to the design-point P_pre_comp.
    P_pre_comp:  Optional[float] = None    # bar

    P_shaft_cases: List[float] = field(default_factory=lambda: [3.075e6])
    P_sweep_min:   float = 0.6e6
    P_sweep_max:   float = 3.0e6
    P_sweep_n:     int   = 30


# =====================================================================
# Dimensional / mass inputs (was DimSizing top-level constants)
# =====================================================================
@dataclass
class DimensionalConfig:
    # ---- HPC (axial, air) ----
    HPC_Inlet_HTR:   float = 0.45
    HPC_Outlet_HTR:  float = 0.70
    HPC_U_tip:       float = 450.0          # m/s
    HPC_Psi:         float = 0.45           # work coefficient
    HPC_BladeChord:  float = 0.03           # m
    HPC_Spacing:     float = 0.3            # gap as fraction of chord
    HPC_M_ax:        float = 0.5            # axial Mach number

    # ---- HPT (axial, combustion products) ----
    HPT_Inlet_HTR:   float = 0.80
    HPT_Hub_Margin:  float = 0.90           # outlet-hub floor = HPC_inlet_hub * margin
    HPT_U_tip:       float = 500.0          # m/s
    HPT_Psi:         float = 1.75
    HPT_BladeChord:  float = 0.04           # m
    HPT_Spacing:     float = 0.3
    HPT_M_ax:        float = 0.3

    # ---- Combustor (RQL) ----
    CC_C_ax:         float = 40.0           # m/s, axial velocity
    CC_tau_rich:     float = 2.0e-3         # s, rich-zone residence
    CC_tau_lean:     float = 3.5e-3         # s, lean-zone residence
    CC_f_quench:     float = 0.20           # quench-air fraction

    # ---- Mass estimation anchors ----
    SP_turboshaft:   float = 10.0e3         # W/kg, GE T408 anchor
    SP_recup:        float = 14.0e3         # W/kg, Microfire recuperator
    system_margin:   float = 0.23           # fraction of bare engine + recup

    # DLR V2500 hot-section mass fractions (Oestreicher et al. 2025)
    m_HPC_kg_ref:    float = 284.0
    m_CC_kg_ref:     float = 151.0
    m_HPT_kg_ref:    float = 191.0


# =====================================================================
# Expander inputs (was ExpanderSizing top-level constants)
# =====================================================================
@dataclass
class ExpanderConfig:
    f_crank:     float = 80.0               # Hz, crankshaft frequency
    N_cyl:       int   = 4                  # cylinders
    sigma_allow: float = 600e6              # Pa, IN718 allowable stress
    Sp_power:    float = 0.2                # kW/kg, assembly specific power
    # Outlet pressure of the GH2 expander (downstream of HEX, into combustor)
    P3_H2:       float = 8.0                # bar


# =====================================================================
# Output / runtime options
# =====================================================================
@dataclass
class OutputConfig:
    csv_path:     str  = "propulsion_results.csv"
    print_report: bool = True
    show_plots:   bool = False              # set True to display plots interactively
    save_plots:   bool = True               # write report-ready PNGs to plots_dir
    plots_dir:    str  = "plots"            # output directory for PNGs
    plot_dpi:     int  = 200                # raster resolution for PNGs


# =====================================================================
# Bundle
# =====================================================================
@dataclass
class Config:
    cycle:     CycleConfig       = field(default_factory=CycleConfig)
    offdesign: OffDesignConfig   = field(default_factory=OffDesignConfig)
    dim:       DimensionalConfig = field(default_factory=DimensionalConfig)
    expander:  ExpanderConfig    = field(default_factory=ExpanderConfig)
    output:    OutputConfig      = field(default_factory=OutputConfig)
