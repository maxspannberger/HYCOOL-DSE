from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
import sys

from rich.table import Table


# Allow running from project root or from inside WeightEstimations
root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.append(str(root))

from WeightEstimations.Aircraft_Config import AircraftConfig, default_q400_hycool
from WeightEstimations.mainClassII import ClassIIResult, run_class_ii
from General.component_parameters import component_params as comp_params


@dataclass
class CgCalculationInput:

    MAC:            float       # m 
    l_f:            float       # fuselage lenght, m

    OEW:            float       # operating empty weight
    MTOW:           float
    W_fixed:        float
    
    W_h2_tank:      float       #tank weight
    L_tank:         float       #tank length

    l_n:            float       #nose lenght
    l_c:            float       #cabin lenght
    l_tc:           float    #tail cone length

    W_fus:          float       #fuselage weight

    W_lg_nose:           float    #nose landing gear weight
    W_lg_main:           float    #main landing gear weight

    W_htail:        float
    W_vtail:        float

    MAC_h:          float
    MAC_v:          float

    W_wing:         float
    W_sc:           float   #surface control system weight

    x_LEMAC:        float           #intital guess x_LEMAC

    W_engine:       float           


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


        L_tank = final["L_tank_m"]

        l_n = cfg.l_n
        l_c = cfg.l_c
        l_tc = cfg.l_tc

        MAC_h = cfg.MAC_h
        MAC_v = cfg.MAC_v

        W_wing = result.weight.W_wing_accurate
        W_sc = result.weight.W_sc

        MAC = result.MAC
        x_LEMAC = cfg.LEMAC

        W_engine = result.weight.W_engine         #total propulsion system weight, excluding lh2 tank but including piping TODO: perhaps exclude piping & cabling for the cg calc


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


            L_tank = L_tank,

            l_n = l_n,
            l_c = l_c,
            l_tc = l_tc,

            MAC_h = MAC_h,
            MAC_v = MAC_v,

            W_wing = W_wing,
            W_sc = W_sc,

            MAC = MAC,
            x_LEMAC = x_LEMAC,

            W_engine = W_engine,

        )


@dataclass
class CgBreakdown:
    #Then the breakdown class should only store calculated values. It should not calculate anything major.
    OEW_cg: float



class CgCalculator:
    """
    Computes total aircraft CG from component masses and x-locations.
    """

    def __init__(self, inputs: CgCalculationInput):
        self.i = inputs
        self._validate()

    def _validate(self) -> None:
        if not self.i.components:
            raise ValueError("No CG components were provided.")

        if self.i.mac_m <= 0:
            raise ValueError("MAC must be positive for x_cg_mac calculation.")

        for comp in self.i.components:
            if comp.mass_kg < 0:
                raise ValueError(f"Negative mass for component: {comp.name}")
            
    @staticmethod
    def cg_from_weights(
        weights: list[float],
        locations: list[float],
    ) -> float:
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

        x_LEMAC = d.x_LEMAC
        MAC = d.MAC

        l_f = d.l_f
        l_n = d.l_n
        l_c = d.l_c
        L_tank = d.L_tank

        MAC_h = d.MAC_h
        MAC_v = d.MAC_v




# ------------------- Fuselage Group Weights (group components according to Torenbeek p.301) ------------------
        W_fixed = d.W_fixed
        W_fus = d.W_fus
        W_lg_nose = d.W_lg_nose
        W_htail = d.W_htail
        W_vtail = d.W_vtail
        W_h2_tank = d.W_h2_tank 

# ------------------- Fuselage Group cg locations  ------------------

        x_cg_fixed = cfg.cg_location_fus * l_f                      #assume cg of fixed weight to be equal to fuselage cg, TODO: could be shifted a bit
        x_cg_fus = cfg.cg_location_fus * l_f
        x_cg_lg_nose = (2/3) * l_c                                  #this is just an estimate, TODO: can be calculated from required load for steering (SEAD)        
        x_cg_htail = 0.98*l_f-MAC_h+(cfg.cg_location_tail_c)        #took 2% fus lenght fort the little cone behind tail, then cg is at a torenbeek defined frn behind LE TODO: update when l_h is updated
        x_cg_vtail = 0.98*l_f-MAC_v+(cfg.cg_location_tail_c)        #took 2% fus lenght fort the little cone behind tail, then cg is at a torenbeek defined frn behind LE
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


        x_cg_sc = x_LEMAC + MAC
        x_cg_wing = x_LEMAC + 
        x_cg_lg_main = x_LEMAC + 0.5 * MAC         #initial estimate from Torenbeek p.301, TODO: to be fixed for cg excursion & tipover angle
        x_cg_power_units = x_LEMAC + cfg.cg_location_engines

        

        return CgBreakdown(
           
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

    inp = CgCalculationInput.from_config(cfg, result)
    estimator = CgCalculator(inp)
    breakdown = estimator.compute()

    print(breakdown.summary())