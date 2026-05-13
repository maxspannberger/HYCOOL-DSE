"""
Aircraft_Config.py

SSoT for all "Need" parameters that flow between Class II
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

    # --- Propulsion Densities -------------------------------------------
    rho_hts_motor:     float = 40
    rho_turbine_core:  float = 10
    turbine_penalty:   float = 1.4
    cryo_penalty:      float = 1.15
    grav_density:      float = 0.64

    # --- Tail sizing targets -------------------------------------------
    V_h_target:       float = 0.95   # HT volume coefficient (jet/turboprop)
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
    c_root      = 4.97                          # Referenced
    b_v_initial = 5.5                           # Referenced
    MAC_v       = 3.5                           # Referenced
    return AircraftConfig(
        # Wing
        S_ref            = 81.68,               # Class I Value
        b                = 28.58,               # Class I Value
        AR               = 10,                  # Class I Value
        MAC              = 2.86,                # Class I Value
        c_root           = c_root,
        tc_root          = 0.12,                # Referenced
        tc_mean          = 0.11,                # Referenced
        sweep_half       = np.deg2rad(23.0),    # Referenced
        sweep_tc         = np.deg2rad(24.0),    # Referenced

        # Horizontal tail
        S_h_initial      = 24,                  # Referenced
        MAC_h            = 2.10,                # Referenced
        tc_h             = 0.12,                # Referenced
        sweep_h_half     = np.deg2rad(22.0),    # Referenced
        sweep_h_tc       = np.deg2rad(20.0),    # Referenced
        l_h              = 17.5,                # Referenced

        # Vertical tail
        MAC_v            = MAC_v,               
        tc_v             = 0.12,                # Referenced
        sweep_v_half     = np.deg2rad(33.0),    # Referenced
        sweep_v_tc       = np.deg2rad(35.0),    # Referenced
        l_v              = 17.5,                # Referenced
        b_v_initial      = b_v_initial,         
        t_tail           = False,               # Design Decision
        h_h              = b_v_initial,
        S_v_initial      = MAC_v * b_v_initial,

        # Fuselage
        l_f              = 36.55,               # Class I Value
        b_f              = 2.9,                 # Class I Value
        h_f              = 2.9,                 # Class I Value
        S_wet_f          = 298.15,              # Class I Value
        l_t              = 17.5,                # Referenced

        # Flight envelope
        altitude_cruise  = 7_620,               # From Mission Definition
        M_cruise         = 0.7,                 # From Mission Definition
        V_cruise         = 0.7 * 296.0,         # From Mission Definition
        V_dive           = 213.5,               # --- TBD ---
        V_stall          = 48.6,                # Class I Value

        # Mission
        range_m          = 1_000_000.0,         # From Mission Definition
        eta_prop         = 0.90,                # Assumed
        eta_thermal      = 0.40,                # Assumed

        altitude_reserve = 457.2,               # 1500ft, standard
        t_reserve        = 45 * 60.0,           # From Mission Definition
        V_climb_EAS      = 130.0,               # Assumed
        ROC_avg          = 7.62,                # Assumed
        TO_taxi_frac     = 0.02,                # Assumed

        m_dot_fuel       = 0.071,               # Initial Assumption

        # Propulsion
        T_TO_per_engine  = 20_000.0,            # Initial Assumption
        y_engine         = 5.0,                 # Assumed
        W_propulsion     = 2_500.0,             # Assumed
        N_engines        = 2,                   # Class I Value
        D_propfan        = 4.0,                 # Assumed
        eta_static_loss  = 0.80,                # Assumed
        eta_prop_V2      = 0.70,                # Assumed
        LD_takeoff       = 11.0,                # Assumed

        # Propulsion Densities
        rho_hts_motor    = 20,                  # Referenced
        rho_turbine_core = 10,                  # Referenced
        turbine_penalty  = 1.4,                 # --- TBD ---
        cryo_penalty     = 1.15,                # --- TBD ---
        grav_density     = 0.64,                # Referenced

        # Tail sizing targets (propfan: V_v bumped from 0.085 to 0.10)
        V_h_target       = 0.95,                # Torenbeek
        V_v_target       = 0.10,                # Torenbeek
        AR_h             = 4.5,                 # Torenbeek
        AR_v             = 1.7,                 # Torenbeek

        # Config
        high_wing        = False,               # Design Decision
        has_flap_slat    = True,                # Design Decision

        # Loads
        n_ult            = 3.75,                # CS-25 Requirements

        # Mission masses
        W_payload        = 10_000.0,            # Class I Value
        W_fixed          = 5_500.0,             # Torenbeek

        # Iteration
        MTOW_initial     = 31_729.92,           # Class I Value
    )