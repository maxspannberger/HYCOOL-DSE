from dataclasses import dataclass

# =============================================================================
# Define the global configuration dataclass for the fluid properties and constants
# =============================================================================
@dataclass
class H2SystemConfig:
    # Global simulation configurations
    fluid: str = 'Hydrogen'
    
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
    cool_dummy_A: float = 1.0                # placeholder cross-sectional area for active cooling [m^2]
    cool_dummy_u: float = 1.0                # placeholder flow velocity for active cooling [m/s]
    cool_dummy_dp: float = 1.0               # placeholder pressure drop for active cooling [Pa]

    # Constants for hts
    eps_hts: float = 0.0015*10**-3           # internal surface roughness for cooling lines [m]
    A_slot: float = 0.0092347                # slot cross sectional area [m^2]
    N_slots: float = 24                      # number of slots of hts components
    L: float = 0.2253                        # stator length
    VF: float = 0.35                         # stator void factor


    # Default baseline geometric parameters for pipe segments
    pipe_default_d: float = 0.02             # default baseline inner pipeline diameter [m]
    pipe_default_segments: int = 10          # default segment mesh resolution density across components
    pipe_default_N: int = 10                 # default baseline number of MLI layers applied
    pipe_default_N_bar: float = 5.5          # default baseline insulation layer density [layers/cm]
    pipe_default_P_mli: float = 10**(-4)     # default baseline residual gas pressure value [Pa]
    pipe_default_eps: float = 2e-6           # default baseline structural pipe inner roughness [m]