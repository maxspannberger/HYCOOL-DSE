"""
Export_Results.py

Dumps every numeric output of the Class II estimation into machine-
readable .csv files and a human-readable .txt summary so anyone can
open the results in Excel / a text editor without running Python.

Files produced (default: ./outputs/):
    class_ii_results.csv      flat key-value dump of every field,
                              grouped by section
    class_ii_results.txt      plain-text version of the same data,
                              laid out in readable sections
    class_ii_iterations.csv   per-iteration MTOW convergence trace
                              (only written when an iteration log is
                              supplied to export_results)
"""

import csv
import os
from dataclasses import fields, is_dataclass
from typing import Any, Iterable


# ---------------------------------------------------------------------
# Field grouping
# ---------------------------------------------------------------------

# A "section" is a list of (label, value) pairs. We build sections from
# each breakdown using dataclasses.fields plus any @property values we
# want to surface (CD0, CD_total, L_over_D, W_empty, m_LH2_total, ...).

def _pairs_from_dataclass(obj: Any) -> list[tuple[str, Any]]:
    """Return [(field_name, value), ...] for a dataclass instance."""
    return [(f.name, getattr(obj, f.name)) for f in fields(obj)]


def _format_value(v: Any) -> str:
    """Render a value for CSV / TXT output."""
    if isinstance(v, bool):
        return "True" if v else "False"
    if isinstance(v, float):
        if v == 0:
            return "0"
        absv = abs(v)
        if absv >= 1e5 or absv < 1e-3:
            return f"{v:.6e}"
        return f"{v:.6g}"
    return str(v)


# ---------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------

def _top_level_section(result) -> list[tuple[str, Any]]:
    return [
        ("MTOW [kg]",            result.MTOW),
        ("MZFW [kg]",            result.MZFW),
        ("W_empty (OEW) [kg]",   result.W_empty),
        ("W_fuel (LH2) [kg]",    result.W_fuel),
        ("W_payload [kg]",       result.W_payload),
        ("W_fixed [kg]",         result.W_fixed),
        ("Cruise L/D [-]",       result.L_over_D),
        ("CL_cruise [-]",        result.CL_cruise),
        ("iterations [-]",       result.iterations),
        ("converged [-]",        result.converged),
    ]


def _tail_section(tail) -> list[tuple[str, Any]]:
    return _pairs_from_dataclass(tail)


def _drag_section(drag) -> list[tuple[str, Any]]:
    base = _pairs_from_dataclass(drag)
    # Add derived properties
    base += [
        ("CD0 (total, with misc factor) [-]", drag.CD0),
        ("CD_lift_dep [-]",                   drag.CD_lift_dep),
        ("CD_total [-]",                      drag.CD_total),
        ("L_over_D [-]",                      drag.L_over_D),
    ]
    return base


def _weight_section(weight) -> list[tuple[str, Any]]:
    base = _pairs_from_dataclass(weight)
    base += [
        ("W_structure [kg]", weight.W_structure),
        ("W_empty (OEW) [kg]", weight.W_empty),
    ]
    return base


def _mission_section(mission) -> list[tuple[str, Any]]:
    base = _pairs_from_dataclass(mission)
    base += [
        ("m_LH2_total [kg]", mission.m_LH2_total),
        ("P_max (shaft, across phases) [W]", mission.P_max),
    ]
    return base


def _power_section(power) -> list[tuple[str, Any]]:
    return _pairs_from_dataclass(power)


def _sections(result) -> list[tuple[str, list[tuple[str, Any]]]]:
    return [
        ("Summary",                       _top_level_section(result)),
        ("Tail Sizing (initial)",         _tail_section(result.tail)),
        ("Tail Sizing (rechecked w/ T_TO)", _tail_section(result.tail_rechecked)),
        ("Drag Breakdown (cruise)",       _drag_section(result.drag)),
        ("Weight Breakdown",              _weight_section(result.weight)),
        ("Mission Power & Fuel",          _mission_section(result.mission)),
        ("Power & Thrust Sizing",         _power_section(result.power)),
    ]


# ---------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------

def write_csv(result, path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Section", "Parameter", "Value"])
        for section_name, pairs in _sections(result):
            for name, val in pairs:
                w.writerow([section_name, name, _format_value(val)])


# ---------------------------------------------------------------------
# TXT writer
# ---------------------------------------------------------------------

def write_txt(result, path: str) -> None:
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("CLASS II INTEGRATED SIZING RESULTS")
    lines.append("=" * 72)
    lines.append("")

    for section_name, pairs in _sections(result):
        lines.append("-" * 72)
        lines.append(section_name)
        lines.append("-" * 72)
        width = max((len(n) for n, _ in pairs), default=0)
        for name, val in pairs:
            lines.append(f"  {name:<{width}}  =  {_format_value(val)}")
        lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------
# Iteration trace (optional)
# ---------------------------------------------------------------------

def write_iterations_csv(iterations: Iterable[dict], path: str) -> None:
    iterations = list(iterations)
    if not iterations:
        return
    keys = list(iterations[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for row in iterations:
            w.writerow([_format_value(row[k]) for k in keys])


# ---------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------

def export_results(
    result,
    output_dir: str = "outputs",
    iterations: Iterable[dict] | None = None,
) -> dict:
    """
    Write CSV and TXT exports of a ClassIIResult. Returns the paths
    written so the caller can echo them to the user.
    """
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, "class_ii_results.csv")
    txt_path = os.path.join(output_dir, "class_ii_results.txt")
    write_csv(result, csv_path)
    write_txt(result, txt_path)

    paths = {"csv": csv_path, "txt": txt_path}

    if iterations is not None:
        it_path = os.path.join(output_dir, "class_ii_iterations.csv")
        write_iterations_csv(iterations, it_path)
        paths["iterations_csv"] = it_path

    return paths
