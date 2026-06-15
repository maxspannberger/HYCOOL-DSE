"""
main.py
=======
Top-level orchestrator for the propulsion sizing chain.

Pipeline
--------
   config.py
      |
      v
   GasTurbineCycle           (TurbineSizing.py)
      |
      +-----> OffDesignEvaluator   (OffDesign.py)
      |
      +-----> DimensionalSizing    (DimSizing.py)
      |
      +-----> PistonExpander       (ExpanderSizing.py)
      |
      v
   propulsion_results.csv  (single wide row + one extra row per off-design case)

Usage
-----
    python main.py

Edit config.py to change inputs. Toggle plots / quietness via
config.output.{show_plots, print_report}.
"""

import csv
from pathlib import Path

from rocketcea.cea_obj_w_units import CEA_Obj

from config         import Config
from TurbineSizing  import GasTurbineCycle
from OffDesign      import OffDesignEvaluator
from DimSizing      import DimensionalSizing
from ExpanderSizing import PistonExpander


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _to_csv_scalar(v):
    """Convert numpy/bool to plain CSV-friendly scalars."""
    try:
        import numpy as np
        if isinstance(v, (np.floating, np.integer)):
            return v.item()
        if isinstance(v, np.bool_):
            return bool(v)
    except ImportError:
        pass
    return v


def _config_inputs_row(cfg):
    """Flatten the input config into a dict of `cfg__section__key` columns."""
    out = {}
    for section_name in ("cycle", "offdesign", "dim", "expander"):
        section = getattr(cfg, section_name)
        for k, v in section.__dict__.items():
            if isinstance(v, (list, tuple, dict)):
                # Skip composite inputs in CSV
                continue
            out[f"cfg__{section_name}__{k}"] = v
    return out


def _write_csv(path, cases):
    """
    Write a vertical CSV: one row per parameter, one column per case.

    Layout:
        section, parameter, <case_1>, <case_2>, ...
        cfg     P_target    2.0e6     2.0e6
        cycle   mdot_f      0.0287    0.0287
        offdesign  TIT_od   <blank>   1869.29
        ...

    `cases` : dict ordered as {case_name: {full_key: value}}.
    Keys keep their `<section>__<name>` prefixes; the prefix is split off
    into the dedicated `section` column so each row stays short.

    Blank cells mean the parameter doesn't exist for that case (e.g. off-design
    parameters in the design row).
    """
    if not cases:
        return

    case_names = list(cases.keys())

    # Stable order: first time a (section, parameter) appears across any case.
    rows_order = []
    seen = set()
    for case_data in cases.values():
        for full_key in case_data.keys():
            if "__" in full_key:
                section, param = full_key.split("__", 1)
            else:
                section, param = "", full_key
            tag = (section, param, full_key)
            if tag not in seen:
                seen.add(tag)
                rows_order.append(tag)

    # Group by section while preserving first-seen order within each section.
    section_order = []
    by_section = {}
    for section, param, full_key in rows_order:
        if section not in by_section:
            by_section[section] = []
            section_order.append(section)
        by_section[section].append((param, full_key))

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["section", "parameter"] + case_names)
        for section in section_order:
            for param, full_key in by_section[section]:
                row = [section, param]
                for case_name in case_names:
                    v = cases[case_name].get(full_key, "")
                    row.append(_to_csv_scalar(v))
                w.writerow(row)
            # Blank line between sections for visual grouping
            w.writerow([])


# ----------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------
def run(P_opt=None, off_design_cases=None, input_conditions=None, cfg=None, show=False, write=False):
    """
    Run the full sizing chain and write a CSV.

    Returns a dict with handles to every sized object so a caller can do
    additional plotting / inspection.
    """
    cfg = cfg or Config()

    cea = CEA_Obj(
        oxName="AIR", fuelName="GH2",
        pressure_units="bar", temperature_units="K", isp_units="sec",
    )

    # Adjust cfg for new inputs
    if P_opt is not None:
        cfg.cycle.P_target = P_opt
    if off_design_cases is not None:
        cfg.offdesign.P_shaft_cases = off_design_cases
    if input_conditions is not None:
        cfg.cycle.P_pre_comp = input_conditions["p"]

    # --- 1. Design-point cycle ---
    engine = GasTurbineCycle.from_config(cfg).size(cea=cea)

    # --- 2. Off-design sweep + headline cases ---
    od_eval = OffDesignEvaluator.from_config(engine, cfg)
    od_cases = {}
    for P in cfg.offdesign.P_shaft_cases:
        try:
            od_cases[P] = od_eval.evaluate(P)
        except ValueError as exc:
            print(f"[warn] OffDesign at {P/1e6:.3f} MW failed: {exc}")
            od_cases[P] = None

    # --- 3. Dimensional sizing ---
    dim = DimensionalSizing.from_config(engine, cfg, cea=cea).size()

    # --- 4. Piston expander ---
    expander = PistonExpander.from_config(engine, cfg).size()

    # --- Reports ---
    if show:
        if cfg.output.print_report:
            engine.report()
            for P, res in od_cases.items():
                if res is not None:
                    od_eval.report(res)
            dim.report()
            expander.report()

        if cfg.output.show_plots:
            engine.plot_ts()
            engine.plot_ts_h2()
            dim.plot()

    # --- Build per-case parameter maps; CSV is written vertically with one
    #     column per case and one row per parameter.
    inputs_row = _config_inputs_row(cfg)

    # Shared across every case: inputs + design-point cycle + dim + expander.
    shared = {}
    shared.update(inputs_row)
    shared.update(engine.to_csv_row())
    shared.update(dim.to_csv_row())
    shared.update(expander.to_csv_row())

    cases = {"design": dict(shared)}
    for P, res in od_cases.items():
        if res is None:
            continue
        col_name = f"offdesign_{P/1e6:.3f}MW"
        case_data = dict(shared)
        case_data.update(OffDesignEvaluator.to_csv_row(res, prefix="offdesign"))
        cases[col_name] = case_data

    if write:
        _write_csv(cfg.output.csv_path, cases)
        print(f"\n[ok] CSV written: {cfg.output.csv_path}  "
            f"({len(cases)} column(s): {', '.join(cases.keys())})")

    return dict(engine=engine, od_eval=od_eval, od_cases=od_cases,
                dim=dim, expander=expander, cases=cases)


# ----------------------------------------------------------------------
# CLI entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    run(show=False, write=False)
