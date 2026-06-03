from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import sys

root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.append(str(root))

@dataclass
class LoadingDiagramInput:
    # ---------------- Masses ----------------
    MTOW: float
    OEW: float
    Fuel_MaxP: float
    MaxCargo: float

    # ---------------- Cargo layout ----------------
    MaxFwdCargoVol: float
    MaxAftCargoVol: float

    # ---------------- Passenger layout ----------------
    PaxWeight: float
    Pax_count: int
    rows: int
    FirstWindow: float
    LastWindow: float

    # ---------------- Longitudinal geometry ----------------
    LEMAC: float
    mac: float

    OEW_CG_global: float
    FUEL_CG_global: float
    AftCargo_global: float
    FwdCargo_global: float

    margin_mac: float = 0.02

    @classmethod
    def from_class_ii(
        # pull values from result, cfg, and aero_dict
        cls,
        cfg,
        result,
        *,
        aero_dict: Optional[dict] = None,
    ) -> "LoadingDiagramInput":

        final = result.iteration_log[-1]

        PaxWeight = float(cfg.PaxWeight)
        Pax_count = int(cfg.Pax_count)
        PaxWeight_Tot = PaxWeight * Pax_count
        MaxCargo = float(cfg.W_payload - PaxWeight_Tot)

        MaxFwdCargoVol = float(cfg.Max_fwd_cargo_vol)
        MaxAftCargoVol = float(cfg.Max_aft_cargo_vol)

        Seats_abreast = cfg.Seats_abreast
        rows = int(np.ceil(Pax_count/Seats_abreast))
        

        return cls(
            # These should come from Class II
            MTOW=float(result.MTOW),
            OEW=float(result.OEW),
            Fuel_MaxP=float(result.W_fuel),

            # These are still aircraft layout assumptions for now
            MaxCargo=MaxCargo,
            MaxFwdCargoVol=MaxFwdCargoVol,
            MaxAftCargoVol=MaxAftCargoVol,

            PaxWeight=PaxWeight,
            Pax_count=Pax_count,
            rows=rows,

            FirstWindow=float(cfg.FirstWindow),
            LastWindow=float(cfg.LastWindow),

            LEMAC=float(cfg.LEMAC),

            # Prefer the final Class II MAC if available
            mac=float(final["MAC_m"]),

            OEW_CG_global=float(cfg.OEW_cg),
            FUEL_CG_global=float(cfg.FUEL_cg),
            AftCargo_global=float(cfg.AftCargo_cg),
            FwdCargo_global=float(cfg.FwdCargo_cg),
        )

@dataclass
class LoadingDiagramBreakdown:
    positions_mac: np.ndarray
    positions_fwd: np.ndarray
    positions_aft: np.ndarray

    up_cg_cargo_fwd: np.ndarray
    up_mass_cargo_fwd: np.ndarray
    up_cg_cargo_aft: np.ndarray
    up_mass_cargo_aft: np.ndarray

    up_cg_win_fwd: np.ndarray
    up_mass_win_fwd: np.ndarray
    up_cg_aisle_fwd: np.ndarray
    up_mass_aisle_fwd: np.ndarray

    up_cg_win_aft: np.ndarray
    up_mass_win_aft: np.ndarray
    up_cg_aisle_aft: np.ndarray
    up_mass_aisle_aft: np.ndarray

    up_cg_fuel: np.ndarray
    up_mass_fuel: np.ndarray

    leftmost_limit: np.ndarray
    rightmost_limit: np.ndarray
    leftmost_limit_margin: np.ndarray
    rightmost_limit_margin: np.ndarray
    lim_y: np.ndarray

    xcg_lower: float
    xcg_upper: float
    xcg_lower_margin: float
    xcg_upper_margin: float


class LoadingDiagramEstimator:
    def __init__(self, inputs: LoadingDiagramInput):
        self.i = inputs
        self._validate()

    def _validate(self) -> None:
        d = self.i
        required = dict(
            MTOW=d.MTOW,
            OEW=d.OEW,
            Fuel_MaxP=d.Fuel_MaxP,
            MaxCargo=d.MaxCargo,
            MaxFwdCargoVol=d.MaxFwdCargoVol,
            MaxAftCargoVol=d.MaxAftCargoVol,
            PaxWeightTot=d.PaxWeightTot,
            Pax_count=d.Pax_count,
            rows=d.rows,
            LEMAC=d.LEMAC,
            mac=d.mac,
        )

        missing = [name for name, value in required.items() if value <= 0]
        if missing:
            raise ValueError(f"Loading diagram inputs not set or zero: {missing}")

    def convert_global(self, global_dim):
        d = self.i
        return (global_dim - d.LEMAC) / d.mac

    def loading_pax(self, positions, cg_in, mass_in):
        d = self.i
        PaxWeight = d.PaxWeight

        updated_cg, updated_mass = [cg_in], [mass_in]
        for pos in positions:
            new_mass = updated_mass[-1] + 2 * PaxWeight
            new_cg = (updated_mass[-1] * updated_cg[-1] + pos * PaxWeight * 2) / new_mass
            updated_cg.append(new_cg)
            updated_mass.append(new_mass)
        return np.array(updated_cg), np.array(updated_mass)

    #Fwd cargo first
    def loading_fwd_cargo(self, cg_in, mass_in):
        d = self.i

        MaxFwdCargo = (
            d.MaxFwdCargoVol
            / (d.MaxFwdCargoVol + d.MaxAftCargoVol)
            * d.MaxCargo
        )

        MaxAftCargo = (
            d.MaxAftCargoVol
            / (d.MaxFwdCargoVol + d.MaxAftCargoVol)
            * d.MaxCargo
        )

        FwdCargo_LEMAC = self.convert_global(d.FwdCargo_global)
        AftCargo_LEMAC = self.convert_global(d.AftCargo_global)

        updated_mass = [mass_in, mass_in + MaxFwdCargo, mass_in + d.MaxCargo]
        updated_cg = [
            cg_in,
            (cg_in * mass_in + MaxFwdCargo * FwdCargo_LEMAC) / (mass_in + MaxFwdCargo),
            (cg_in * mass_in + MaxFwdCargo * FwdCargo_LEMAC + MaxAftCargo * AftCargo_LEMAC) / (mass_in + d.MaxCargo),
        ]
        return np.array(updated_cg), np.array(updated_mass)

    #Aft cargo first
    def loading_aft_cargo(self, cg_in, mass_in):
        d = self.i

        MaxFwdCargo = (
            d.MaxFwdCargoVol
            / (d.MaxFwdCargoVol + d.MaxAftCargoVol)
            * d.MaxCargo
        )
        MaxAftCargo = (
            d.MaxAftCargoVol
            / (d.MaxFwdCargoVol + d.MaxAftCargoVol)
            * d.MaxCargo
        )

        FwdCargo_LEMAC = self.convert_global(d.FwdCargo_global)
        AftCargo_LEMAC = self.convert_global(d.AftCargo_global)

        updated_mass = [mass_in, mass_in + MaxAftCargo, mass_in + d.MaxCargo]
        updated_cg = [
            cg_in,
            (cg_in * mass_in + MaxAftCargo * AftCargo_LEMAC) / (mass_in + MaxAftCargo),
            (
                cg_in * mass_in
                + MaxFwdCargo * FwdCargo_LEMAC
                + MaxAftCargo * AftCargo_LEMAC
            ) / (mass_in + d.MaxCargo),
        ]

        return np.array(updated_cg), np.array(updated_mass)

    def loading_fuel(self, cg_in, mass_in):
        d = self.i
        FUEL_CG_LEMAC = self.convert_global(d.FUEL_CG_global)

        new_mass = mass_in + d.Fuel_MaxP
        new_cg = (mass_in * cg_in + FUEL_CG_LEMAC * d.Fuel_MaxP) / new_mass

        return np.array([cg_in, new_cg]), np.array([mass_in, new_mass])

    def compute(self) -> LoadingDiagramBreakdown:
        d = self.i

        OEW_CG_LEMAC = self.convert_global(d.OEW_CG_global)

        positions_mac = self.convert_global(
            np.linspace(d.FirstWindow, d.LastWindow, d.rows)
        )
        positions_fwd = positions_mac
        positions_aft = np.flip(positions_mac)

        # Both Fwd -> Aft and Aft -> Fwd Cargo Paths
        up_cg_cargo_fwd, up_mass_cargo_fwd = self.loading_fwd_cargo(OEW_CG_LEMAC, d.OEW)
        up_cg_cargo_aft, up_mass_cargo_aft = self.loading_aft_cargo(OEW_CG_LEMAC, d.OEW)

        # Fwd -> Aft Path
        up_cg_win_fwd, up_mass_win_fwd = self.loading_pax(
            positions_fwd,
            up_cg_cargo_fwd[-1],
            up_mass_cargo_fwd[-1],
        )
        up_cg_aisle_fwd, up_mass_aisle_fwd = self.loading_pax(
            positions_fwd,
            up_cg_win_fwd[-1],
            up_mass_win_fwd[-1],
        )

        # Aft -> Fwd Path
        up_cg_win_aft, up_mass_win_aft = self.loading_pax(
            positions_aft,
            up_cg_cargo_aft[-1],
            up_mass_cargo_aft[-1],
        )
        up_cg_aisle_aft, up_mass_aisle_aft = self.loading_pax(
            positions_aft,
            up_cg_win_aft[-1],
            up_mass_win_aft[-1],
        )

        # Final Fuel Loading
        up_cg_fuel, up_mass_fuel = self.loading_fuel(
            up_cg_aisle_fwd[-1],
            up_mass_aisle_fwd[-1],
        )

        leftmost_limit = np.min(up_cg_aisle_fwd) * np.ones(2)
        rightmost_limit = np.max(up_cg_cargo_aft) * np.ones(2)
        lim_y = np.linspace(d.OEW, d.MTOW, 2)

        leftmost_limit_margin = leftmost_limit - d.margin_mac
        rightmost_limit_margin = rightmost_limit + d.margin_mac

        return LoadingDiagramBreakdown(
            positions_mac=positions_mac,
            positions_fwd=positions_fwd,
            positions_aft=positions_aft,

            up_cg_cargo_fwd=up_cg_cargo_fwd,
            up_mass_cargo_fwd=up_mass_cargo_fwd,
            up_cg_cargo_aft=up_cg_cargo_aft,
            up_mass_cargo_aft=up_mass_cargo_aft,

            up_cg_win_fwd=up_cg_win_fwd,
            up_mass_win_fwd=up_mass_win_fwd,
            up_cg_aisle_fwd=up_cg_aisle_fwd,
            up_mass_aisle_fwd=up_mass_aisle_fwd,

            up_cg_win_aft=up_cg_win_aft,
            up_mass_win_aft=up_mass_win_aft,
            up_cg_aisle_aft=up_cg_aisle_aft,
            up_mass_aisle_aft=up_mass_aisle_aft,

            up_cg_fuel=up_cg_fuel,
            up_mass_fuel=up_mass_fuel,

            leftmost_limit=leftmost_limit,
            rightmost_limit=rightmost_limit,
            leftmost_limit_margin=leftmost_limit_margin,
            rightmost_limit_margin=rightmost_limit_margin,
            lim_y=lim_y,

            xcg_lower=float(leftmost_limit[0]),
            xcg_upper=float(rightmost_limit[0]),
            xcg_lower_margin=float(leftmost_limit_margin[0]),
            xcg_upper_margin=float(rightmost_limit_margin[0]),
        )

    def plot(
        self,
        breakdown: Optional[LoadingDiagramBreakdown] = None,
        *,
        show: bool = True,
        save_path: Optional[str | Path] = None,
    ) -> LoadingDiagramBreakdown:
        bd = breakdown if breakdown is not None else self.compute()

        plt.figure(figsize=(8, 6))

        path_fwd = {"marker": "o", "markersize": 4, "linewidth": 1.5}
        path_aft = {"marker": "o", "markersize": 4, "linewidth": 1.5, "linestyle": "--"}

        plt.plot(bd.up_cg_cargo_aft, bd.up_mass_cargo_aft, label="Cargo (Aft-Fwd)", **path_aft)
        plt.plot(bd.up_cg_cargo_fwd, bd.up_mass_cargo_fwd, label="Cargo (Fwd-Aft)", **path_fwd)
        plt.plot(bd.up_cg_win_aft, bd.up_mass_win_aft, label="Window Pax (Aft-Fwd)", **path_aft)
        plt.plot(bd.up_cg_win_fwd, bd.up_mass_win_fwd, label="Window Pax (Fwd-Aft)", **path_fwd)
        plt.plot(bd.up_cg_aisle_aft, bd.up_mass_aisle_aft, label="Aisle Pax (Aft-Fwd)", **path_aft)
        plt.plot(bd.up_cg_aisle_fwd, bd.up_mass_aisle_fwd, label="Aisle Pax (Fwd-Aft)", **path_fwd)
        plt.plot(bd.up_cg_fuel, bd.up_mass_fuel, label="Fuel Loading", marker="D", lw=2)

        plt.plot(bd.leftmost_limit, bd.lim_y, label="Stability Limit", color="black", **path_fwd)
        plt.plot(bd.rightmost_limit, bd.lim_y, color="black", **path_fwd)
        plt.plot(
            bd.leftmost_limit_margin,
            bd.lim_y,
            label=r"Stability Limit (2$\%$ LEMAC Margin)",
            color="black",
            **path_aft,
        )
        plt.plot(bd.rightmost_limit_margin, bd.lim_y, color="black", **path_aft)

        plt.title("Aircraft Loading Diagram (Potato Plot)", fontsize=14, fontweight="bold")
        plt.xlabel(r"$x_{cg}$ [$\%$ MAC]", fontsize=12)
        plt.ylabel("Mass [kg]", fontsize=12)
        plt.xlim(0, 1)
        plt.legend(title="Loading Sequence", loc="upper right", frameon=True, shadow=True)

        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close()

        return bd
    
if __name__ == "__main__":
    from WeightEstimations.Aircraft_Config import default_q400_hycool
    from WeightEstimations.mainClassII import run_class_ii
    from General.component_parameters import component_params as comp_params

    cfg = default_q400_hycool()
    result = run_class_ii(cfg, comp=comp_params, tol=1.0, max_iter=100, verbose=True)

    inp = LoadingDiagramInput.from_class_ii(cfg, result)
    estimator = LoadingDiagramEstimator(inp)

    breakdown = estimator.compute()
    estimator.plot(breakdown, show=True, save_path="loading_diagram.png")

    print(f"CG lower without margin: {breakdown.xcg_lower:.3f}")
    print(f"CG upper without margin: {breakdown.xcg_upper:.3f}")
    print(f"CG lower with margin: {breakdown.xcg_lower_margin:.3f}")
    print(f"CG upper with margin: {breakdown.xcg_upper_margin:.3f}")