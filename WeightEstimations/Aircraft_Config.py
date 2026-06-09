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
    Loading:        float
    taper:          float          # lambda = c_tip / c_root
    CL_max:         float          # maximum lift coefficient, clean configuration
    CL_alpha0_clean: float         # lift curve slope, clean configuration
    Cm_alpha0_clean: float         # moment curve slope, clean configuration
    alpha_CL0_clean: float         # angle of attack for zero lift, clean configuration
    t_c_flap:       float          # flap thickness-to-chord ratio, assumed for now

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

    l_n:            float       #nose lenght
    l_c:            float       #cabin lenght
    l_tc:         float       #tail cone length

    # --- H2 tank --------------------------------------------------------
    # Hump-back: tank rides on top of fuselage (747-style); fuselage
    # length is unchanged and only the wetted area grows. Otherwise
    # the tank sits in-line and lengthens the fuselage.
    hump_back:      bool
    rho_LH2_eff:    float          # kg/m^3, effective LH2 density in tank

    # --- Flight envelope -----------------------------------------------
    altitude_cruise: float
    M_cruise:        float
    V_cruise:        float
    V_cruise_EAS:    float
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
    y_engine_2:       float = 0.0
    y_engine_4:       float = 0.0
    W_propulsion:     float = 0.0
    N_engines:        int   = 2
    N_propellers:      int   = 4
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
    grav_density_std:  float = 0.05 # TODO: adjust

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
    W_fixed_frn:      float = 0.0

    # --- Iteration control ---------------------------------------------
    MTOW_initial:     float = 0.0

    # --- Layout & Passengers ---------------------------------------------
    PaxWeight:          float = 0.0
    Pax_count:          int = 0
    Max_fwd_cargo_vol:  float = 0.0                   
    Max_aft_cargo_vol:  float = 0.0         
    Seats_abreast:      int = 0       
    LEMAC:              float = 0.0
    lfn:                float = 0.0
    hh:                 float = 0.0

    FirstWindow:        float = 0.0
    LastWindow:         float = 0.0

    OEW_cg:             float = 0.0
    FUEL_cg:            float = 0.0
    AftCargo_cg:        float = 0.0
    FwdCargo_cg:        float = 0.0

    #--- cg_breakdown --------------------------------------------------
    x_cg_htail:         float = 0.0
    x_cg_vtail:         float = 0.0
    x_cg_fus:           float = 0.0
    x_cg_lg_nose:       float = 0.0
    x_cg_lg_main:       float = 0.0
    x_cg_sc:            float = 0.0
    x_cg_engine:        float = 0.0

    cg_location_fus:     float = 0.0                    # distance from fus. nose to fus cg, as % of fus length
    cg_location_tail_c:  float = 0.0                    # % of chord from LE 
    cg_location_tail_b:  float = 0.0                    # % of semi-span from root chord
    cg_surf_control:     float = 0.0                    # 100% of MAC from LEMAC
    cg_location_engines: float = 0.0                    # [m] distance from LEMAC to cg of power units on wings
    
    OEW_target_rel:      float = 0.0

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
    WingLoading_Target = 3810                   # Class I Value
    ClassI_MTOW = 31_729.92                     # Class I Value
    b_init                = 28.58,               # Class I Value
    return AircraftConfig(
        # Wing
        S_ref            = ClassI_MTOW * 9.80665 / WingLoading_Target,
        b                = 28.58,               # Class I Value
        AR               = 10,                  # Class I Value
        MAC              = 2.86,                # Class I Value
        c_root           = c_root,
        tc_root          = 0.10,                # Referenced
        tc_mean          = 0.11,                # Referenced
        sweep_half       = np.deg2rad(23.0),    # Referenced
        sweep_tc         = np.deg2rad(24.0),    # Referenced
        Loading          = WingLoading_Target,  # Class I Value
        taper            = 0.4,                 # lambda, typical transport

        #aerodynamic values for chosen airfoil mix, 64A410 for the root and SC(2)-0612 for the tip
        CL_max=1.50,                             # taken from diagram, clean configuration
        CL_alpha0_clean=0.265,                   # according to XFLR5 data for the chosen airfoil
        Cm_alpha0_clean=-0.265,                   # according to XFLR5 data for the chosen airfoil
        alpha_CL0_clean=-3.24*np.pi/180,                       # according to XFLR5 data for the chosen airfoil
        t_c_flap=0.12,                # Assumed, typical for transport

        # Horizontal tail
        S_h_initial      = 24,                  # Referenced
        MAC_h            = 2.10,                # Referenced
        tc_h             = 0.12,                # Referenced
        sweep_h_half     = np.deg2rad(22.0),    # Referenced
        sweep_h_tc       = np.deg2rad(20.0),    # Referenced
        l_h              = 20.6,                # Referenced

        # Vertical tail
        MAC_v            = MAC_v,               
        tc_v             = 0.12,                # Referenced
        sweep_v_half     = np.deg2rad(33.0),    # Referenced
        sweep_v_tc       = np.deg2rad(35.0),    # Referenced
        l_v              = 19.7,                # Referenced
        b_v_initial      = b_v_initial,         
        t_tail           = False,               # Design Decision
        h_h              = b_v_initial,
        S_v_initial      = MAC_v * b_v_initial,

        # Fuselage
        l_f              = 35.05,               # Class I Value
        b_f              = 2.9,                 # Class I Value
        h_f              = 2.9,                 # Class I Value
        S_wet_f          = 298.15,              # Class I Value
        l_t              = 17.5,                # Referenced

        l_n              = 5.08,                # nose lenght, from class I
        l_c              = 22,                  # cabin lenght, from class I
        l_tc             = 7.98,                # tail cone length, from class I

        # H2 tank
        hump_back        = True,               # Design Decision
        rho_LH2_eff      = 70.85,               # kg/m^3, LH2 at boiling point

        # Flight envelope
        altitude_cruise  = 6_096,               # From Mission Definition 7_620 old was FL250 
        M_cruise         = 0.68,                # From Mission Definition
        V_cruise         = 0.68 * 316,          # From Mission Definition 309.7 old for FL250
        V_cruise_EAS     = 140.9706457,         # Equivalent cruise speed, check scissor plot excel for calc
        V_dive           = 179.7978853,         # from CS25 CS 25.335, check scissor plot excel, 176.2133072 old

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
        y_engine_2       = 7.0,                 # Assumed
        y_engine_4       = 5.0,                 # Assumed
        W_propulsion     = 2_500.0,             # Assumed
        N_engines        = 2,                   # Class I Value
        N_propellers      = 4,                   # Class I Value
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
        W_fixed_frn      = 0.14,                # Torenbeek p. 287 14% of MTOW

        # Iteration
        MTOW_initial     = ClassI_MTOW,         # Class I Value

        #Layout & Passengers
        PaxWeight           = 84,                   # EASA
        Pax_count           = 100,                  # Requirement
        Max_fwd_cargo_vol   = 6,                    # Fwd cargo hold volume, placeholder for now
        Max_aft_cargo_vol   = 4,                    # Aft cargo hold volume, placeholder for now
        Seats_abreast       = 4,                    # Class I

        FirstWindow         = 6.74,                 # Distance nose tip to first window [m], placeholder for now
        LastWindow          = 27.08,                # Distance nose tip to last window [m], placeholder for now

        LEMAC               = 15.4,                 # Distance nose tip to LEMAC [m], placeholder for now
        lfn                 = 14.35,                 # Distance nose tip to LE wing root LEMAC - 1.05
        hh                  = 4,                    # Normal distance from wing plane to tail plane, placeholder for now

        OEW_cg              = 16.98,                # Distance nose tip to OEW CG [m], placeholder for now
        FUEL_cg             = 28.3,                   # Distance nose tip to Fuel CG [m], placeholder for now
        AftCargo_cg         = 25,                   # Distance nose tip to Aft Cargo CG [m], placeholder for now
        FwdCargo_cg         = 9,                   # Distance nose tip to Fwd Cargo CG [m], placeholder for now
    
        #cg Breakdown - VERY rough guesses for now
        x_cg_htail          = 30,
        x_cg_vtail          = 30,
        x_cg_fus            = 18,
        x_cg_lg_nose        = 5,
        x_cg_lg_main        = 14,
        x_cg_sc             = 15,
        x_cg_engine         = 12.5,

        # torenbeek cg estimation values, table 8-15 p.294
        cg_location_fus     = 0.41,                 # distance from fus. nose to fus cg, as % of fus length
        cg_location_tail_c  = 0.42,                 # % of chord from LE at 0.38 span
        cg_location_tail_b  = 0.38,                 # % of semi-span from root chord
        cg_surf_control     = 1,                    # 100% of MAC from LEMAC

        cg_location_engines = 0.5,                  # [m] from LEMAC to cg of the power units on the wing
        OEW_target_rel      = 0.5,                 # % of MAC, from LEMAC. Value for config 3 (wing mtd engines) from Torenbeek p.300 (range is 0.2-0.25)
    
    )