from dataclasses import dataclass, field
from typing import List

# =============================================================================
# Define the global configuration dataclass for the fluid properties and constants
# =============================================================================
@dataclass
class H2SystemConfig:
    # Global simulation configurations
    fluid: str = 'Hydrogen'                  # FLUID used for simulation
    max_error: float = 100                    # SOLVER convergence treshhold
    tank_p: float = 200000                   # TANK pressure [Pa]
    tank_T: float = 20.3                     # TANK temperature [T]
    tank_d: float = 0.012                    # TANK outlet diameter [m]
    T_amb: float = 317                       # AMBIENT temperature [K]
    
    # ------------------------------
    normal_phases: List[str]   = field(default_factory=lambda: ['TO', 'climb', 'cruise', 'APP'])
    normal_m_dots: List[float] = field(default_factory=lambda: [0.0865, 0.0895, 0.0669, 0.0206])  
    
    oei_phases: List[str]      = field(default_factory=lambda: ['OEI_gt', 'OEI_mot', 'OEI_bus'])
    oei_m_dots: List[float]    = field(default_factory=lambda:  [0.0490, 0.0484, 0.0542]) 
    # ------------------------------

    # Solver configuration parameters
    divergence_penalty: float = 1e9          # penalty value if the numerical solver diverges
    tank_max_gas_frac: float = 0.01          # gas fraction limit before the incompressibility assumption breaks
    
    # MLI insulation properties (Lockheed constants)
    pipe_mli_cs: float = 1.93 * 10**-6       # MLI conductivity coefficient [W/(m*K^(3.63))]
    pipe_mli_cr: float = 3.88 * 10**-10      # MLI radiation coefficient [W/(m^2*K^(4.67))]
    pipe_mli_cg: float = 5.5 * 10**4         # MLI gas conduction coefficient [W/(m^2*Torr*K^(0.52))]
    pipe_mli_eps: float = 0.03               # MLI emissivity, typical value for aluminized Mylar
    
    # Component setup placeholders
    tank_initial_u: float = 0.0              # flow velocity inside the storage tank [m/s]
    cool_dummy_dp: float = 1000.0            # placeholder pressure drop for active cooling [Pa]

    # Constants for hts
    eps_hts: float = 0.0015*10**-3           # internal surface roughness for cooling lines [m]
    A_slot: float = 0.0092347                # slot cross sectional area [m^2]
    N_slots: float = 24                      # number of slots of hts components
    L: float = 0.2253                        # stator length
    VF: float = 0.35                         # stator void factor

    # Default baseline geometric parameters for pipe segments
    pipe_default_d: float = 0.012            # default baseline inner pipeline diameter [m]
    pipe_segment_length: float = 0.1        # default baseline segment length [m]
    pipe_default_N: int = 10                 # default baseline number of MLI layers applied
    pipe_default_N_bar: float = 5.5          # default baseline insulation layer density [layers/cm]
    pipe_default_P_mli: float = 10**(-4)     # default baseline residual gas pressure value [Torr]
    pipe_default_eps: float = 2e-6           # default baseline structural pipe inner roughness [m]
    # pipe_default_eps: float = 3.5e-3
    
    # Pump parameters
    pump_efficiency: float = 0.75            # default pump isentropic efficiency
    pump_electric_efficiency: float = 0.9    # general efficiency of the pump's electric motor (for power calculations)

    operating_temp = {
        "hts_gen": 35.0,
        "ac_dc": 250.0,
        "bus": 250.0,
        "dc_ac": 250.0,
        "hts_pow": 35.0,
    }

    HEX_default_d = 0.009
    HEX_effectiveness = 0.6
    HEX_extra_thickness = 0.005
    initial_length_scaling = 50

    k_Al = 167      # W/m K
    k_TMI_293 = 0.194    # W/m K
    k_TMI_4 = 0.095 # W/m K
    t_TMI = 0.0001 # m
    
    HTS_default_d = 0.009
    HTS_channels = 1

    FPI_relaxation = 0.5    # relaxation for fixed point iteration to stabilize convergence

    # corner loss factor
    corner_loss_factor: float = 1.4         # factor to overestimate pressure losses in corner instead of underestimating