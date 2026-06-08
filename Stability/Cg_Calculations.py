from __future__ import annotations

from pathlib import Path
import sys
from dataclasses import dataclass
from rich.table import Table

from rich.console import Console



# Allow running from project root or from inside WeightEstimations
root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.append(str(root))

from WeightEstimations.Aircraft_Config import AircraftConfig, default_q400_hycool
from WeightEstimations.mainClassII import ClassIIResult, run_class_ii
from General.component_parameters import component_params as comp_params


@dataclass
class CgCalculationInput:
    l_f:            float

    OEW:            float       # operating empty weight
    MTOW:           float
    W_fixed:        float

    W_fus:          float
    W_lg_nose:      float   #nose landing gear weight
    W_lg_main:          float    #main landing gear weight
    W_htail:          float 
    W_vtail:          float 
    W_h2_tank:          float 

    cg_location_fus:    float
    cg_location_tail_c: float
    cg_location_engines:    float

    L_tank:         float       #tank length

    l_n:            float       #nose lenght
    l_c:            float       #cabin lenght
    l_tc:           float    #tail cone length

    MAC_h:          float
    MAC_v:          float
    MAC:            float       # m 
    x_LEMAC:        float           #intital guess x_LEMAC


    W_wing:         float
    W_sc:           float   #surface control system weight
    W_engine:       float

    location_wing_cg: float 

    OEW_target_rel: float




    @classmethod
    def from_class_ii(
        cls,
        cfg: AircraftConfig,
        result: ClassIIResult,
    ) -> "CgCalculationInput":
        
        final = result.iteration_log[-1]
        l_f = result.l_f_m

        OEW = result.OEW
        MTOW = result.MTOW
        W_fixed = result.W_fixed

        W_fus = result.weight.W_fus
        W_lg_nose = result.weight.W_lg_nose
        W_lg_main = result.weight.W_lg_main
        W_htail = result.weight.W_htail
        W_vtail = result.weight.W_vtail
        W_h2_tank = result.weight.W_h2_tank

        cg_location_fus = cfg.cg_location_fus 
        cg_location_tail_c = cfg.cg_location_tail_c
        cg_location_engines = result.distance_le_mac_to_turbine       # [m] from LEMAC to cg of the power units on the wing

        L_tank = final["L_tank_m"]

        l_n = cfg.l_n
        l_c = cfg.l_c
        l_tc = cfg.l_tc

        MAC_h = cfg.MAC_h
        MAC_v = cfg.MAC_v

        MAC = result.MAC
        x_LEMAC = cfg.LEMAC

        W_wing = result.weight.W_wing_accurate
        W_sc = result.weight.W_sc
        W_engine = result.weight.W_engine         #total propulsion system weight, excluding lh2 tank but including piping TODO: perhaps exclude piping & cabling for the cg calc


        location_wing_cg = result.distance_le_mac_to_cg         #[m] distance from LEMAC to wing cg 

        OEW_target_rel = cfg.OEW_target_rel                     # statistically determined factor from torenbeek: % of MAC for OEW cg

        return cls(
            l_f = l_f,
            OEW = OEW,
            MTOW = MTOW,
            W_fixed = W_fixed,

            W_fus = W_fus,
            W_lg_nose = W_lg_nose,
            W_lg_main = W_lg_main,
            W_htail = W_htail,
            W_vtail = W_vtail,
            W_h2_tank = W_h2_tank,

            cg_location_fus = cg_location_fus, 
            cg_location_tail_c = cg_location_tail_c,
            cg_location_engines = cg_location_engines,

            L_tank = L_tank,

            l_n = l_n,
            l_c = l_c,
            l_tc = l_tc,

            MAC_h = MAC_h,
            MAC_v = MAC_v,

            W_wing = W_wing,
            W_sc = W_sc,
            W_engine = W_engine,

            MAC = MAC,
            x_LEMAC = x_LEMAC,

            location_wing_cg = location_wing_cg,

            OEW_target_rel = OEW_target_rel,
        )


@dataclass
class CgBreakdown:
    #Then the breakdown class should only store calculated values. It should not calculate anything major.
    OEW_check: float
    OEW_excl_fixed: float
    x_cg_OEW: float
    W_wing_group: float
    W_fus_group: float
    x_cg_fus_group: float
    x_cg_wing_group: float
    X_LEMAC_new: float
    l_h : float


    def summary(self) -> Table:
        table = Table(
            title="CG Calculation Breakdown",
            show_header=True,
            header_style="bold blue",
        )

        table.add_column("Group")
        table.add_column("Mass [kg]", justify="right")
        table.add_column("x CG [m]", justify="right")

        table.add_row(
            "Fuselage group",
            f"{self.W_fus_group:.1f}",
            f"{self.x_cg_fus_group:.2f}",
        )

        table.add_row(
            "Wing group",
            f"{self.W_wing_group:.1f}",
            f"{self.x_cg_wing_group:.2f}",
        )

        table.add_section()

        table.add_row(
            "[bold]OEW check[/bold]",
            f"[bold]{self.OEW_check:.1f}[/bold]",
            f"[bold]{self.x_cg_OEW:.2f}[/bold]",
        )

        table.add_row(
            "[bold]OEW check excl W_fixed[/bold]",
            f"[bold]{self.OEW_excl_fixed:.1f}[/bold]",
        )

        table.add_row(
            "[bold]X_LEMAC[/bold]",
            f"[bold]{self.X_LEMAC_new:.1f}[/bold]",
        )

        table.add_row(
            "[bold]Tail Length Estimate (l_h)[/bold]",
            f"[bold]{self.l_h:.1f}[/bold]",
        )

        return table



class CgCalculator:
    """
    Computes total aircraft CG from component masses and x-locations.
    """

    def __init__(self, inputs: CgCalculationInput):
        self.i = inputs
        self._validate()

    def _validate(self) -> None:
        if self.i.MAC <= 0:
            raise ValueError("MAC must be positive.")

        if self.i.l_f <= 0:
            raise ValueError("Fuselage length must be positive.")

        masses = [
            self.i.OEW,
            self.i.MTOW,
            self.i.W_fixed,
            self.i.W_h2_tank,
            self.i.W_fus,
            self.i.W_lg_nose,
            self.i.W_lg_main,
            self.i.W_htail,
            self.i.W_vtail,
            self.i.W_wing,
            self.i.W_sc,
            self.i.W_engine,
        ]

        if any(mass < 0 for mass in masses):
            raise ValueError("All masses must be non-negative.")
            
    @staticmethod
    def cg_from_weights(
        weights: list[float],
        locations: list[float],
    ) -> tuple[float, float]:
        """
        Calculate the CG location from multiple weights and x-locations.
        """

        if len(weights) != len(locations):
            raise ValueError("weights and locations must have the same length.")

        total_weight = sum(weights)

        if total_weight <= 0.0:
            raise ValueError("Total weight must be positive for CG calculation.")

        total_moment = sum(
            W * x for W, x in zip(weights, locations)
        )

        cg_location = total_moment / total_weight

        return total_weight, cg_location

    def compute(self) -> CgBreakdown:
        d = self.i

        #x_LEMAC = d.x_LEMAC
        MAC = d.MAC

        l_f = d.l_f
        l_n = d.l_n
        l_c = d.l_c
        L_tank = d.L_tank

        MAC_h = d.MAC_h
        MAC_v = d.MAC_v

        cg_location_fus = d.cg_location_fus 
        cg_location_tail_c = d.cg_location_tail_c
        cg_location_engines = d.cg_location_engines




# ------------------- Fuselage Group Weights (group components according to Torenbeek p.301) ------------------
        W_fixed = d.W_fixed
        W_fus = d.W_fus
        W_lg_nose = d.W_lg_nose
        W_htail = d.W_htail
        W_vtail = d.W_vtail
        W_h2_tank = d.W_h2_tank 

# ------------------- Fuselage Group cg locations  ------------------

        x_cg_fixed = cg_location_fus * l_f                      #assume cg of fixed weight to be equal to fuselage cg, TODO: could be shifted a bit
        x_cg_fus = cg_location_fus * l_f
        x_cg_lg_nose = (2/3) * l_n                                  #this is just an estimate, TODO: can be calculated from required load for steering (SEAD)        
        x_cg_htail = 0.98*l_f-MAC_h+(cg_location_tail_c*MAC_h)        #took 2% fus lenght fort the little cone behind tail, then cg is at a torenbeek defined frn behind LE TODO: update when l_h is updated
        x_cg_vtail = 0.98*l_f-MAC_v+(cg_location_tail_c*MAC_v)        #took 2% fus lenght fort the little cone behind tail, then cg is at a torenbeek defined frn behind LE
        x_cg_tank = l_n + l_c + 1/2 * L_tank

        
        W_fus_group, x_cg_fus_group = self.cg_from_weights(
            weights=[
                W_fixed,
                W_fus,
                W_lg_nose,
                W_htail,
                W_vtail,
                W_h2_tank,
            ],
            locations=[
                x_cg_fixed,
                x_cg_fus,
                x_cg_lg_nose,
                x_cg_htail,
                x_cg_vtail,
                x_cg_tank,
            ],
        )

# ------------------- Wing Group ------------------
        W_sc = d.W_sc
        W_lg_main = d.W_lg_main
        W_wing = d.W_wing
        W_engine = d.W_engine

        location_wing_cg = d.location_wing_cg

        W_wing_group, x_cg_wing_group_rel = self.cg_from_weights(
            weights=[
                W_sc,
                W_lg_main,
                W_wing,
                W_engine,
            ],
            locations=[
                MAC,
                0.5 * MAC,
                location_wing_cg,
                cg_location_engines,
            ],
        )

        x_cg_OEW_rel_target = d.OEW_target_rel * MAC

        x_LEMAC = x_cg_fus_group - x_cg_OEW_rel_target + (W_wing_group/W_fus_group) * (x_cg_wing_group_rel - x_cg_OEW_rel_target)
        #x_LEMAC = 17

        x_cg_sc = x_LEMAC + MAC
        x_cg_lg_main = x_LEMAC + 0.5 * MAC         #initial estimate from Torenbeek p.301, TODO: to be fixed for cg excursion & tipover angle
        x_cg_wing = x_LEMAC + location_wing_cg
        x_cg_power_units = x_LEMAC + cg_location_engines

        W_wing_group, x_cg_wing_group = self.cg_from_weights(
            weights=[
                W_sc,
                W_lg_main,
                W_wing,
                W_engine,
            ],
            locations=[
                x_cg_sc,
                x_cg_lg_main,
                x_cg_wing,
                x_cg_power_units,
            ]
        )

        OEW_check = W_fus_group + W_wing_group
        OEW_excl_fixed = OEW_check - W_fixed
        x_cg_OEW = ((W_fus_group*x_cg_fus_group + W_wing_group*x_cg_wing_group)/(W_fus_group + W_wing_group))

        l_h = l_f - (x_LEMAC - 1/4 * MAC) - 0.02*l_f - 3/4 * MAC_h      #TODO here I assumed a 2% of fus length for little cone behind tail considered l_h distance 1/4c wing to 1/4 horizontal tail. 

        return CgBreakdown(
            OEW_check=OEW_check,
            OEW_excl_fixed=OEW_excl_fixed,
            x_cg_OEW=x_cg_OEW,
            W_wing_group=W_wing_group,
            W_fus_group=W_fus_group,
            x_cg_fus_group=x_cg_fus_group,
            x_cg_wing_group=x_cg_wing_group,
            X_LEMAC_new = x_LEMAC,
            l_h = l_h,
        )


if __name__ == "__main__":
    cfg = default_q400_hycool()

    result = run_class_ii(
        cfg,
        comp=comp_params,
        tol=1.0,
        max_iter=100,
        verbose=False,
        config=3,
    )



    inp = CgCalculationInput.from_class_ii(cfg, result)
    estimator = CgCalculator(inp)
    breakdown = estimator.compute()

    console = Console()
    console.print(breakdown.summary())
