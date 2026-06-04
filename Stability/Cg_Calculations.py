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
class CgComponent:
    """
    Single mass item for CG calculation.

    x_location is measured from the chosen reference point.
    Recommended reference: nose tip, because your current config already
    stores several CG positions from the nose.
    """

    name: str
    mass_kg: float
    x_location_m: float
    group: str = "General"

    @property
    def moment_kgm(self) -> float:
        return self.mass_kg * self.x_location_m


@dataclass
class CgCalculationInput:
    """
    Input container for the CG calculation.

    All component masses and positions are collected here before computing
    total mass and aircraft CG.
    """

    reference: str
    components: list[CgComponent] = field(default_factory=list)

    mac_m: float = 0.0
    lemac_m: float = 0.0

    @classmethod
    def from_config(
        cls,
        cfg: AircraftConfig,
        result: Optional[ClassIIResult] = None,
    ) -> "CgCalculationInput":
        """
        Build CG input from AircraftConfig and, where useful, ClassIIResult.

        Use cfg for user-defined component positions.
        Use result for final computed masses such as fuel mass, OEW, propulsion mass, etc.
        """

        components: list[CgComponent] = []

        # ------------------------------------------------------------
        # Final mass values
        # ------------------------------------------------------------
        # If result is available, use final Class II masses.
        # Otherwise use rough config/default values.
        if result is not None:
            OEW = float(result.OEW)
            fuel_mass = float(result.W_fuel)
            payload_mass = float(result.W_payload)
            prop_mass = float(result.W_prop)

            if result.iteration_log:
                final = result.iteration_log[-1]
                mac_m = float(final["MAC_m"])
            else:
                mac_m = float(cfg.MAC)

        else:
            OEW = float(cfg.MTOW_initial - cfg.W_payload)
            fuel_mass = float(cfg.m_dot_fuel * cfg.range_m / cfg.V_cruise)
            payload_mass = float(cfg.W_payload)
            prop_mass = float(cfg.W_propulsion)
            mac_m = float(cfg.MAC)

        # ------------------------------------------------------------
        # Basic aircraft groups
        # ------------------------------------------------------------
        components.append(
            CgComponent(
                name="Operating empty weight",
                mass_kg=OEW,
                x_location_m=float(cfg.OEW_cg),
                group="Aircraft"
            )
        )

        components.append(
            CgComponent(
                name="Fuel",
                mass_kg=fuel_mass,
                x_location_m=float(cfg.FUEL_cg),
                group="Fuel"
            )
        )

        components.append(
            CgComponent(
                name="Propulsion system",
                mass_kg=prop_mass,
                x_location_m=float(getattr(cfg, "Propulsion_cg", cfg.OEW_cg)),
                group="Propulsion"
            )
        )

        # ------------------------------------------------------------
        # Payload split into passengers and cargo
        # ------------------------------------------------------------
        pax_mass = float(cfg.PaxWeight * cfg.Pax_count)
        cargo_mass = max(payload_mass - pax_mass, 0.0)

        # Simple passenger CG estimate:
        # average between first and last window.
        pax_cg = 0.5 * (float(cfg.FirstWindow) + float(cfg.LastWindow))

        components.append(
            CgComponent(
                name="Passengers",
                mass_kg=pax_mass,
                x_location_m=pax_cg,
                group="Payload"
            )
        )

        # Cargo split based on available cargo volume
        total_cargo_vol = float(cfg.Max_fwd_cargo_vol + cfg.Max_aft_cargo_vol)

        if total_cargo_vol > 0.0 and cargo_mass > 0.0:
            fwd_cargo_mass = cargo_mass * cfg.Max_fwd_cargo_vol / total_cargo_vol
            aft_cargo_mass = cargo_mass * cfg.Max_aft_cargo_vol / total_cargo_vol
        else:
            fwd_cargo_mass = 0.0
            aft_cargo_mass = 0.0

        components.append(
            CgComponent(
                name="Forward cargo",
                mass_kg=fwd_cargo_mass,
                x_location_m=float(cfg.FwdCargo_cg),
                group="Payload"
            )
        )

        components.append(
            CgComponent(
                name="Aft cargo",
                mass_kg=aft_cargo_mass,
                x_location_m=float(cfg.AftCargo_cg),
                group="Payload"
            )
        )

        return cls(
            reference="nose tip",
            components=components,
            mac_m=mac_m,
            lemac_m=float(cfg.LEMAC),
        )


@dataclass
class CgBreakdown:
    """
    Output of the CG calculation.
    """

    total_mass_kg: float
    total_moment_kgm: float
    x_cg_m: float
    x_cg_mac: float
    components: list[CgComponent]

    def summary(self) -> Table:
        table = Table(
            title="Aircraft CG Calculation",
            show_header=True,
            header_style="bold blue"
        )

        table.add_column("Group")
        table.add_column("Component")
        table.add_column("Mass [kg]", justify="right")
        table.add_column("x-location [m]", justify="right")
        table.add_column("Moment [kg m]", justify="right")

        for comp in self.components:
            table.add_row(
                comp.group,
                comp.name,
                f"{comp.mass_kg:.1f}",
                f"{comp.x_location_m:.2f}",
                f"{comp.moment_kgm:.1f}",
            )

        table.add_section()
        table.add_row(
            "[bold]Total[/bold]",
            "",
            f"[bold]{self.total_mass_kg:.1f}[/bold]",
            f"[bold]{self.x_cg_m:.2f}[/bold]",
            f"[bold]{self.total_moment_kgm:.1f}[/bold]",
        )

        table.add_row(
            "[bold green]CG wrt MAC[/bold green]",
            "",
            "",
            "",
            f"[bold green]{self.x_cg_mac:.4f} MAC[/bold green]",
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
        if not self.i.components:
            raise ValueError("No CG components were provided.")

        if self.i.mac_m <= 0:
            raise ValueError("MAC must be positive for x_cg_mac calculation.")

        for comp in self.i.components:
            if comp.mass_kg < 0:
                raise ValueError(f"Negative mass for component: {comp.name}")

    def compute(self) -> CgBreakdown:
        total_mass = sum(comp.mass_kg for comp in self.i.components)

        if total_mass <= 0:
            raise ValueError("Total mass must be positive.")

        total_moment = sum(comp.moment_kgm for comp in self.i.components)

        x_cg = total_moment / total_mass

        # CG relative to MAC:
        # x_cg_mac = 0 means at LEMAC.
        # x_cg_mac = 0.25 means 25 percent MAC behind LEMAC.
        x_cg_mac = (x_cg - self.i.lemac_m) / self.i.mac_m

        return CgBreakdown(
            total_mass_kg=total_mass,
            total_moment_kgm=total_moment,
            x_cg_m=x_cg,
            x_cg_mac=x_cg_mac,
            components=self.i.components,
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