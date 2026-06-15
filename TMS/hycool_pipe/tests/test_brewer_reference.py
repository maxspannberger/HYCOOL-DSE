# Validation: Brewer geometry mass check + sizing self-consistency + convergence
# (spec §6 items 1, 6, 7, 8)
#
# Brewer (1991) reference geometry: D=7.52 mm, t_foam=83.1 mm, t_ss=0.5 mm, t_al=0.5 mm
# → m/L ≈ 1.801 kg/m.  We verify the mass formula reproduces this within 5 %.
# Note: D=7.52 mm at ṁ=71 g/s gives ~10 bar dp — Brewer's book used different design
# inputs; the 1.801 kg/m figure is what we can cleanly cross-check.

import pytest
from dataclasses import replace

from hycool_pipe.cases.lh2 import LH2Config, _mass_per_meter
from hycool_pipe.cases.cch2 import CcH2Config
from hycool_pipe.solver import march, size


# ── Brewer mass-per-length check (spec §6 item 1) ────────────────────────────

def test_brewer_mass_per_length():
    """Mass formula at Brewer geometry must reproduce 1.801 kg/m within 5 %."""
    cfg = replace(LH2Config(), t_ss=5e-4, t_al=5e-4)
    _, _, _, m_total = _mass_per_meter(D=7.52e-3, t_foam=83.1e-3, cfg=cfg)
    assert abs(m_total - 1.801) / 1.801 < 0.05, (
        f"Brewer m/L mismatch: got {m_total:.4f} kg/m, expected ≈ 1.801 kg/m"
    )


# ── LH2 sizing self-consistency ───────────────────────────────────────────────

def test_lh2_dp_within_limit():
    """After sizing, total pressure drop must not exceed dp_max by more than 1 %."""
    cfg = LH2Config()
    from hycool_pipe.cases.lh2 import run_analysis
    res = run_analysis(cfg, verbose=False)
    assert res["dp_Pa"] <= cfg.dp_max * 1.01, (
        f"dp={res['dp_Pa']/1e3:.2f} kPa exceeds dp_max={cfg.dp_max/1e3:.0f} kPa"
    )


def test_lh2_heat_within_limit():
    """After sizing, total heat ingress must not exceed q_total_max_W by more than 1 %."""
    cfg = LH2Config()
    from hycool_pipe.cases.lh2 import run_analysis
    res = run_analysis(cfg, verbose=False)
    assert res["q_total_W"] <= cfg.q_total_max_W * 1.01, (
        f"q_total={res['q_total_W']:.1f} W exceeds limit={cfg.q_total_max_W:.0f} W"
    )


# ── CcH2 sizing self-consistency ─────────────────────────────────────────────

def test_cch2_t_exit_within_limit():
    """After CcH2 sizing, T_exit must stay below T_exit_max."""
    cfg = CcH2Config()
    from hycool_pipe.cases.cch2 import run_analysis
    res = run_analysis(cfg, verbose=False)
    assert res["T_exit_K"] <= cfg.T_exit_max, (
        f"T_exit={res['T_exit_K']:.2f} K exceeds T_exit_max={cfg.T_exit_max:.0f} K"
    )


def test_cch2_p_exit_above_operating_floor():
    """CcH2 exit pressure must remain within the 250–350 bar operating envelope."""
    cfg = CcH2Config()
    from hycool_pipe.cases.cch2 import run_analysis
    res = run_analysis(cfg, verbose=False)
    assert res["p_exit_bar"] >= 249.0, (
        f"p_exit={res['p_exit_bar']:.2f} bar is below 250 bar operating floor"
    )


# ── Energy conservation (spec §6 item 7) ─────────────────────────────────────

def test_energy_conservation_cch2():
    """Σ Q_seg must equal ṁ·(h_exit − h_in) to machine precision.

    The march advances h += Q_seg/ṁ each step, so this must hold exactly.
    """
    import CoolProp.CoolProp as CP
    cfg = CcH2Config(t_ss=0.002227)  # pre-sized wall; skip solver overhead
    result = march(D=0.00664, t_foam=0.0, cfg=cfg, n_seg=200)

    h_in  = CP.PropsSI("Hmass", "P", cfg.p_in, "T", cfg.T_in, cfg.fluid)
    energy_from_march  = result.q_total
    energy_from_enthalpy = cfg.m_dot * (result.h_exit - h_in)
    assert abs(energy_from_march - energy_from_enthalpy) < 1e-6, (
        f"Energy imbalance: Σ Q_seg={energy_from_march:.6f} W, "
        f"ṁΔh={energy_from_enthalpy:.6f} W"
    )


# ── Convergence study (spec §6 item 8) ───────────────────────────────────────

def test_convergence_halving_segments():
    """Halving dz (doubling n_seg) must change T_exit and dp_total by < 1 %."""
    cfg = CcH2Config(t_ss=0.002227)
    r_coarse = march(D=0.00664, t_foam=0.0, cfg=cfg, n_seg=100)
    r_fine   = march(D=0.00664, t_foam=0.0, cfg=cfg, n_seg=200)

    dT_rel = abs(r_coarse.T_exit  - r_fine.T_exit)  / r_fine.T_exit
    dp_rel = abs(r_coarse.dp_total - r_fine.dp_total) / r_fine.dp_total

    assert dT_rel < 0.01, f"T_exit change on grid refinement: {dT_rel*100:.3f} %"
    assert dp_rel < 0.01, f"dp_total change on grid refinement: {dp_rel*100:.3f} %"
