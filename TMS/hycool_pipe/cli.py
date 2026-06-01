# CLI entry point: python -m hycool_pipe lh2 ... / cch2 ...
# See §7 of HYCOOL_pipe_design_spec.md for full argument list.
#
# Usage examples:
#   python -m hycool_pipe lh2 --m-dot 0.071 --L 32 --dp-max 0.5e5 --output lh2.json --plot
#   python -m hycool_pipe cch2 --p-in 350e5 --T-in 50 --sf 2.5 --output cch2.json --plot

import argparse
import json
import sys
from dataclasses import replace


# ── Argument parsers ──────────────────────────────────────────────────────────

def _add_common_flags(p: argparse.ArgumentParser) -> None:
    """Output / display flags shared by both subcommands."""
    p.add_argument("--output", metavar="FILE",
                   help="Write results to FILE as JSON (e.g. results.json)")
    p.add_argument("--plot", action="store_true",
                   help="Save p(z), T(z), x(z), Q(z) plots alongside --output")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress printed sizing summary")


def _build_parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="python -m hycool_pipe",
        description="HYCOOL cryogenic hydrogen pipe sizing tool.",
    )
    sub = root.add_subparsers(dest="case", required=True)

    # ── lh2 ──────────────────────────────────────────────────────────────────
    lh2 = sub.add_parser("lh2", help="Size the LH2 (sub-critical) pipe.")
    lh2.add_argument("--m-dot",       type=float, metavar="KG_S",
                     help="Mass flow rate [kg/s]  (default 0.071)")
    lh2.add_argument("--L",           type=float, metavar="M",
                     help="Pipe length [m]  (default 32)")
    lh2.add_argument("--n-bends",     type=int,   metavar="N",
                     help="Number of bends  (default 10)")
    lh2.add_argument("--p-in",        type=float, metavar="PA",
                     help="Inlet pressure [Pa]  (default 2e5 = 2 bar)")
    lh2.add_argument("--T-in",        type=float, metavar="K",
                     help="Inlet temperature [K]  (default 20)")
    lh2.add_argument("--dp-max",      type=float, metavar="PA",
                     help="Max pressure drop [Pa]  (default 0.5e5 = 0.5 bar)")
    lh2.add_argument("--q-total-max", type=float, metavar="W",
                     help="Max total heat ingress [W]  (default 2000)")
    lh2.add_argument("--T-ambient",   type=float, metavar="K",
                     help="Ambient temperature [K]  (default 315 = hot-ground)")
    lh2.add_argument("--k-foam-base", type=float, metavar="W_MK",
                     help="Base PUF conductivity [W/mK]  (default 0.0173)")
    lh2.add_argument("--foam-aging",  type=float, metavar="FACTOR",
                     help="Foam aging multiplier  (default 1.3)")
    lh2.add_argument("--t-ss",        type=float, metavar="M",
                     help="SS inner wall thickness [m]  (default 3e-4 = 0.3 mm)")
    lh2.add_argument("--t-al",        type=float, metavar="M",
                     help="Al outer jacket thickness [m]  (default 3e-4 = 0.3 mm)")
    lh2.add_argument("--roughness",   type=float, metavar="M",
                     help="Pipe roughness [m]  (default 1.5e-6)")
    lh2.add_argument("--fluid",       type=str,
                     help="CoolProp fluid name  (default 'hydrogen')")
    _add_common_flags(lh2)

    # ── cch2 ─────────────────────────────────────────────────────────────────
    ch2 = sub.add_parser("cch2", help="Size the CcH2 (supercritical) pipe.")
    ch2.add_argument("--m-dot",       type=float, metavar="KG_S",
                     help="Mass flow rate [kg/s]  (default 0.071)")
    ch2.add_argument("--L",           type=float, metavar="M",
                     help="Pipe length [m]  (default 32)")
    ch2.add_argument("--n-bends",     type=int,   metavar="N",
                     help="Number of bends  (default 10)")
    ch2.add_argument("--p-in",        type=float, metavar="PA",
                     help="Inlet pressure [Pa]  (default 350e5 = 350 bar)")
    ch2.add_argument("--T-in",        type=float, metavar="K",
                     help="Inlet temperature [K]  (default 50)")
    ch2.add_argument("--dp-max",      type=float, metavar="PA",
                     help="Max pressure drop [Pa]  (default 100e5 = 100 bar)")
    ch2.add_argument("--T-exit-max",  type=float, metavar="K",
                     help="Max exit temperature [K]  (default 110)")
    ch2.add_argument("--sf",          type=float, metavar="FACTOR",
                     help="Safety factor for hoop stress  (default 2.5)")
    ch2.add_argument("--h2-exposure", type=str,   choices=["gaseous", "liquid"],
                     help="H2 embrittlement mode  (default 'gaseous')")
    ch2.add_argument("--T-ambient",   type=float, metavar="K",
                     help="Ambient temperature [K]  (default 315)")
    ch2.add_argument("--roughness",   type=float, metavar="M",
                     help="Pipe roughness [m]  (default 1.5e-6)")
    ch2.add_argument("--fluid",       type=str,
                     help="CoolProp fluid name  (default 'hydrogen')")
    _add_common_flags(ch2)

    return root


# ── Config builders ───────────────────────────────────────────────────────────

def _build_lh2_cfg(args):
    from hycool_pipe.cases.lh2 import LH2Config
    cfg = LH2Config()
    updates = {}
    if args.m_dot       is not None: updates["m_dot"]            = args.m_dot
    if args.L           is not None: updates["L"]                = args.L
    if args.n_bends     is not None: updates["n_bends"]          = args.n_bends
    if args.p_in        is not None: updates["p_in"]             = args.p_in
    if args.T_in        is not None: updates["T_in"]             = args.T_in
    if args.dp_max      is not None: updates["dp_max"]           = args.dp_max
    if args.q_total_max is not None: updates["q_total_max_W"]    = args.q_total_max
    if args.T_ambient   is not None: updates["T_ambient"]        = args.T_ambient
    if args.k_foam_base is not None: updates["k_foam_base"]      = args.k_foam_base
    if args.foam_aging  is not None: updates["foam_aging_factor"]= args.foam_aging
    if args.t_ss        is not None: updates["t_ss"]             = args.t_ss
    if args.t_al        is not None: updates["t_al"]             = args.t_al
    if args.roughness   is not None: updates["roughness"]        = args.roughness
    if args.fluid       is not None: updates["fluid"]            = args.fluid
    return replace(cfg, **updates)


def _build_cch2_cfg(args):
    from hycool_pipe.cases.cch2 import CcH2Config
    cfg = CcH2Config()
    updates = {}
    if args.m_dot       is not None: updates["m_dot"]         = args.m_dot
    if args.L           is not None: updates["L"]             = args.L
    if args.n_bends     is not None: updates["n_bends"]       = args.n_bends
    if args.p_in        is not None: updates["p_in"]          = args.p_in
    if args.T_in        is not None: updates["T_in"]          = args.T_in
    if args.dp_max      is not None: updates["dp_max"]        = args.dp_max
    if args.T_exit_max  is not None: updates["T_exit_max"]    = args.T_exit_max
    if args.sf          is not None: updates["safety_factor"] = args.sf
    if args.h2_exposure is not None: updates["h2_exposure"]   = args.h2_exposure
    if args.T_ambient   is not None: updates["T_ambient"]     = args.T_ambient
    if args.roughness   is not None: updates["roughness"]     = args.roughness
    if args.fluid       is not None: updates["fluid"]         = args.fluid
    return replace(cfg, **updates)


# ── JSON output ───────────────────────────────────────────────────────────────

def _save_json(result: dict, path: str) -> None:
    """Serialise result dict to JSON.  Converts any non-serialisable values to str."""
    def _make_serialisable(v):
        if isinstance(v, (int, float, str, bool, type(None))):
            return v
        try:
            return float(v)
        except (TypeError, ValueError):
            return str(v)

    clean = {k: _make_serialisable(v) for k, v in result.items()}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(clean, fh, indent=2)
    print(f"Results written to {path}")


# ── Plotting ──────────────────────────────────────────────────────────────────

def _get_march_for_plot(result: dict, cfg, case: str):
    """Run a final march() with the sized geometry to get per-segment arrays."""
    from hycool_pipe.solver import march
    D      = result["D_m"]
    t_foam = result.get("t_foam_m", 0.0)
    t_ss   = result["t_ss_m"]
    cfg_sized = replace(cfg, t_ss=t_ss)
    return march(D=D, t_foam=t_foam, cfg=cfg_sized, n_seg=200)


def _plot_march(march_result, result: dict, case: str, out_path: str) -> None:
    """Save a 4-panel figure: p(z), T(z), x(z), Q_seg(z)."""
    try:
        import matplotlib
        matplotlib.use("Agg")          # non-interactive backend — no display required
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        print("matplotlib not installed — skipping plot (pip install matplotlib).",
              file=sys.stderr)
        return

    z    = march_result.z
    p    = [v / 1e5 for v in march_result.p]   # Pa → bar
    T    = march_result.T
    x    = march_result.x
    Q    = march_result.Q_seg

    # Colour-code by flow regime
    regime_colour = {
        "subcooled_liquid":    "#2196F3",   # blue
        "saturated_two_phase": "#FF9800",   # orange
        "superheated_vapour":  "#F44336",   # red
        "single_phase":        "#2196F3",   # blue (CcH2)
    }

    fig, axes = plt.subplots(4, 1, figsize=(9, 11), sharex=True)
    fig.suptitle(f"HYCOOL pipe sizing — {case.upper()}", fontsize=13, fontweight="bold")

    def _scatter(ax, ydata, ylabel, yunit):
        colours = [regime_colour.get(r, "#2196F3") for r in march_result.regime]
        ax.scatter(z, ydata, c=colours, s=4, linewidths=0)
        ax.set_ylabel(f"{ylabel}  [{yunit}]")
        ax.grid(True, lw=0.4, alpha=0.5)

    _scatter(axes[0], p, "Pressure", "bar")
    _scatter(axes[1], T, "Temperature", "K")

    # Quality panel: shade two-phase region
    axes[2].axhspan(0, 1, color="#FF9800", alpha=0.08, label="two-phase")
    _scatter(axes[2], x, "Quality x", "–")
    axes[2].set_ylim(-1.1, 2.1)
    axes[2].axhline(0, color="grey", lw=0.6, ls="--")
    axes[2].axhline(1, color="grey", lw=0.6, ls="--")

    _scatter(axes[3], Q, "Heat per segment", "W")

    axes[-1].set_xlabel("Axial position  z  [m]")

    # Key scalars as text box
    D_mm  = result["D_mm"]
    t_ss  = result.get("t_ss_mm", result.get("t_ss_m", 0.0) * 1e3)
    OD    = result.get("OD_mm",   result.get("r_o_al_m", 0.0) * 2e3)
    mtot  = result["total_mass_kg"]
    dp    = result.get("dp_total_kPa", result.get("dp_kPa", 0.0))
    info  = (f"D={D_mm:.1f} mm  t_ss={t_ss:.2f} mm  OD={OD:.1f} mm\n"
             f"m_total={mtot:.2f} kg   dp={dp:.1f} kPa")
    fig.text(0.5, 0.01, info, ha="center", fontsize=9,
             bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="grey", lw=0.8))

    plt.tight_layout(rect=[0, 0.04, 1, 0.97])

    # Derive plot filename from output path (replace extension) or use a default
    if out_path:
        plot_path = out_path.rsplit(".", 1)[0] + ".png"
    else:
        plot_path = f"hycool_{case}.png"
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved to {plot_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(argv=None) -> None:
    parser = _build_parser()

    # No subcommand given — show help instead of an error
    if argv is None and len(sys.argv) == 1:
        parser.print_help()
        return

    args = parser.parse_args(argv)
    verbose = not args.quiet

    if args.case == "lh2":
        from hycool_pipe.cases.lh2 import run_analysis
        cfg    = _build_lh2_cfg(args)
        result = run_analysis(cfg, verbose=verbose)

    elif args.case == "cch2":
        from hycool_pipe.cases.cch2 import run_analysis
        cfg    = _build_cch2_cfg(args)
        result = run_analysis(cfg, verbose=verbose)

    else:
        parser.print_help()
        sys.exit(1)

    if args.output:
        _save_json(result, args.output)

    if args.plot:
        march_result = _get_march_for_plot(result, cfg, args.case)
        _plot_march(march_result, result, args.case, args.output)


if __name__ == "__main__":
    main()
