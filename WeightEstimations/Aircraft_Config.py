"""
Aircraft_Config.py

Single source of truth for all "Need" parameters that flow between Class II
modules. Everything that more than one module reads lives here.

Convention:
    - SI units unless noted.
    - Angles stored as radians.
    - Weights stored as MASS in kg. Forces (Newtons) computed where needed.
"""

import numpy as np
from dataclasses import dataclass, field


@dataclass
class AircraftConfig:

    # --- Wing -----------------------------------------------------------
    S_ref:          float
    b:              float
    AR:             float
    MAC:            float
    c_root:         float
    tc_root:        float
    tc_mean:        float
    sweep_half:     float
    sweep_tc:       float

    # --- Horizontal tail -----------------------------------------------
    S_h_initial:    float
    MAC_h:          float
    tc_h:           float
    sweep_h_half:   float
    sweep_h_tc:     float
    l_h:            float

    # --- Vertical tail -------------------------------------------------
    S_v_initial:    float
    MAC_v:          float
    tc_v:           float
    sweep_v_half:   float
    sweep_v_tc:     float
    l_v:            float
    b_v_initial:    float
    t_tail:         bool
    h_h:            float

    # --- Fuselage -------------------------------------------------------
    l_f:            float
    b_f:            float
    h_f:            float
    S_wet_f:        float
    l_t:            float

    # --- Flight envelope -----------------------------------------------
    altitude_cruise: float
    M_cruise:        float
    V_cruise:        float
    V_dive:          float
    V_stall:         float

    # --- Mission --------------------------------------------------------
    range_m:         float
    eta_prop:        float
    eta_thermal:     float

    altitude_reserve: float = 457.2
    t_reserve:        float = 2700.0
    V_climb_EAS:      float = 130.0
    ROC_avg:          float = 7.62
    TO_taxi_frac:     float = 0.02

    LHV_fuel:         float = 120e6

    m_dot_fuel:       float = 0.0
    fuel_reserve_frac: float = 0.05

    # --- Propulsion -----------------------------------------------------
    T_TO_per_engine:  float = 0.0
    y_engine:         float = 0.0
    W_propulsion:     float = 0.0
    N_engines:        int   = 2
    D_propfan:        float = 4.0
    eta_static_loss:  float = 0.80
    eta_prop_V2:      float = 0.70
    LD_takeoff:       float = 11.0

    # --- Tail sizing targets -------------------------------------------
    V_h_target:       float = 1.00   # HT volume coefficient (jet/turboprop)
    V_v_target:       float = 0.10   # VT volume coefficient (propfan ~0.10-0.12)
    AR_h:             float = 4.5    # HT aspect ratio (drives b_h from S_h)
    AR_v:             float = 1.7    # VT aspect ratio (drives b_v from S_v)

    # --- Configuration flags --------------------------------------------
    high_wing:        bool = False
    has_flap_slat:    bool = True

    # --- Loads & limits -------------------------------------------------
    n_ult:            float = 3.75

    # --- Mission masses -------------------------------------------------
    W_payload:        float = 0.0
    W_fixed:          float = 0.0

    # --- Iteration control ---------------------------------------------
    MTOW_initial:     float = 0.0


    # ---------- Derived helpers ---------------------------------------
    @property
    def d_f(self) -> float:
        return 0.5 * (self.b_f + self.h_f)

    @property
    def t_root_abs(self) -> float:
        return self.tc_root * self.c_root

    @property
    def t_cruise(self) -> float:
        return self.range_m / self.V_cruise


def default_q400_hycool() -> AircraftConfig:
    c_root      = 2.54
    b_v_initial = 8.3
    MAC_v       = 3.02
    return AircraftConfig(
        # Wing
        S_ref            = 63.1,
        b                = 28.4,
        AR               = 12.78,
        MAC              = 2.49,
        c_root           = c_root,
        tc_root          = 0.18,
        tc_mean          = 0.15,
        sweep_half       = np.deg2rad(20.0),
        sweep_tc         = np.deg2rad(20.0),

        # Horizontal tail
        S_h_initial      = 13.94,
        MAC_h            = 2.80,
        tc_h             = 0.12,
        sweep_h_half     = np.deg2rad(10.0),
        sweep_h_tc       = np.deg2rad(8.0),
        l_h              = 15.5,

        # Vertical tail
        MAC_v            = MAC_v,
        tc_v             = 0.12,
        sweep_v_half     = np.deg2rad(35.0),
        sweep_v_tc       = np.deg2rad(33.0),
        l_v              = 15.0,
        b_v_initial      = b_v_initial,
        t_tail           = True,
        h_h              = b_v_initial,
        S_v_initial      = MAC_v * b_v_initial,

        # Fuselage
        l_f              = 32.8,
        b_f              = 2.69,
        h_f              = 2.80,
        S_wet_f          = 240.0,
        l_t              = 15.5,

        # Flight envelope
        altitude_cruise  = 7_620,
        M_cruise         = 0.7,
        V_cruise         = 0.7 * 296.0,
        V_dive           = 213.5,
        V_stall          = 55.0,

        # Mission
        range_m          = 1_000_000.0,
        eta_prop         = 0.90,
        eta_thermal      = 0.40,

        altitude_reserve = 457.2,
        t_reserve        = 45 * 60.0,
        V_climb_EAS      = 130.0,
        ROC_avg          = 7.62,
        TO_taxi_frac     = 0.02,

        m_dot_fuel       = 0.071,

        # Propulsion
        T_TO_per_engine  = 20_000.0,
        y_engine         = 5.0,
        W_propulsion     = 2_500.0,
        N_engines        = 2,
        D_propfan        = 4.0,
        eta_static_loss  = 0.80,
        eta_prop_V2      = 0.70,
        LD_takeoff       = 11.0,

        # Tail sizing targets (propfan: V_v bumped from 0.085 to 0.10)
        V_h_target       = 1.00,
        V_v_target       = 0.10,
        AR_h             = 4.5,
        AR_v             = 1.7,

        # Config
        high_wing        = True,
        has_flap_slat    = True,

        # Loads
        n_ult            = 3.75,

        # Mission masses
        W_payload        = 10_000.0,
        W_fixed          = 5_500.0,

        # Iteration
        MTOW_initial     = 31_237.0,
    )