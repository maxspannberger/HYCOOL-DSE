# HYCOOL Pipe Design — Code Handoff Spec

Hand this to Claude Code as the brief. It contains: (1) current state, (2) numbered requirements for the LH₂ case, (3) numbered requirements for the CcH₂ (cryo-compressed) case, (4) target architecture, (5) validation checks, (6) references.

The starting point is `pipe_python.py` (uploaded by the user). The design context is the HYCOOL midterm report, §5.2.7 (Piping), which sizes the LH₂ line via G. D. Brewer's method.

---

## 0. Context

- Aircraft: hydrogen-powered, hydrogen-cooled, preliminary design, ~2045 EIS, two parallel cryogenic feed lines from tank to power systems.
- Reference design point from the report: ṁ = 71 g/s per line, L ≈ 32 m, target Δp = 1 bar, target boil-off ≤ 1 % of ṁ.
- Two storage architectures must be supported by the code: **LH₂** (sub-critical, ~1–3 bar, 20 K) and **CcH₂** (supercritical, 250–350 bar, 25–110 K).
- Inner pipe SS-316L, outer jacket Al-6061, polyurethane foam between. Vacuum-jacket variant is out of scope for now but the architecture should not preclude adding it.

---

## 1. Issues in the current `pipe_python.py`

Fix all of these. They are bugs or hard-coded values that must become variables/iterates.

1. `f = 0.013` hard-coded. Replace with Colebrook–White (or Haaland) iterated on Re(D, ṁ, μ(T,p)).
2. `rho = 71` hard-coded. Replace with CoolProp `PropsSI` (parahydrogen) evaluated per segment.
3. Inner pipe wall t_ss = 0.5 mm and outer jacket t_al = 0.5 mm are asserted, not derived. Add a structural-sizing routine that solves for t_ss from hoop stress at design pressure × safety factor (≥ 2.5 for crewed aircraft) and a minimum manufacturing thickness floor (~0.3 mm SS, ~0.5 mm Al). t_al must additionally resist external buckling under 1 bar Δp if the foam is the only standoff.
4. `k_insulation = 0.0173` is the room-temperature value. Replace with k_PUF(T̄) where T̄ is the log-mean foam temperature per segment; apply an aging factor of 1.3 to account for moisture / cryopumping.
5. The `boil_off_percent_per_m` and `boil_off_percent_total` lines use the same denominator `h_lat * m_dot / 100` but `q_dot` is W/m while `q_total` is W. The per-m line is dimensionally wrong. Pick one definition and rename clearly: `boil_off_frac_per_m` (1/m) vs `boil_off_frac_total` (–).
6. `t_ambient = 295` is sea-level standard. The sizing case must be the **hot ground/idle** case (~315 K), not the cruise case. Add an ambient profile (ground hot, climb, cruise cold ~216 K) and size against worst point.
7. `selected_diameter` and `selected_insulation` are inputs. They should be **outputs** of an iterative solver that satisfies the design constraints.
8. Convective film resistances are implicitly zero. Add (1/h_inner) and (1/h_outer) to the thermal resistance network. h_outer is altitude-dependent natural convection; h_inner depends on flow regime (forced-convection liquid, nucleate boiling, post-CHF, etc.).
9. Radiation and conduction through supports/fittings are neglected. Add a parasitic-conduction lumped term Q̇_parasitic = N_supports × (kA/L)_support × ΔT, and a global plumbing-margin factor (1.3–1.5×) on total heat leak, per NASA's "+ Additional Fuel System Component = 1.5(Tank + Insulation)".
10. Length L = 32 m assumes "half-fuselage + half-wingspan". Apply a routing factor 1.2–1.4 to account for real path.
11. No fluid property lookup — add CoolProp throughout.
12. No phase tracking — the line will go LH₂ → two-phase → GH₂. See §3.
13. No mass-flow margin — add a sizing margin (typically 1.1×) on ṁ for transients.

---

## 2. LH₂ requirements (sub-critical case)

The line starts as subcooled or saturated LH₂ and must reach the engine inlet at a specified state (typically superheated GH₂ at T_exit ≥ ~250 K, x = 1). Boil-off "≤ 1 %" is replaced by:

- x(L) = 1 (fully vaporised at the heat-exchanger inlet, *not* at the engine inlet itself — vaporisation is intentional).
- T_exit ≥ T_exit_min (configurable, e.g. 250 K).
- Δp_total ≤ Δp_max (configurable, e.g. 1 bar with explicit check that Δp_max < p_in − p_sat(T_local) everywhere upstream of the planned vaporisation point if subcooled liquid is required there).
- Maximum quality at any chosen "must-be-liquid" station (e.g. cryo-pump inlet) ≤ x_max (configurable, typically 0).

### 2.1 Property model
- CoolProp fluid: `"parahydrogen"` (or `"ParaHydrogen"`); offer `"hydrogen"` (normal H₂) as a switch.
- Query in two-phase region by (p, h), never (p, T). T is non-unique inside the dome.
- Allow ortho-para heat release as an optional `q_op_release(T)` term per segment (700 J/g at 20 K vs latent 446 J/g). Off by default for equilibrium-para feedstock; on for normal-H₂ feedstock.

### 2.2 Segmented marching scheme
- Discretise pipe into N segments (default 200).
- For each segment z_i → z_{i+1}:
  - Get state at z_i from (p_i, h_i).
  - Compute regime: subcooled liquid / saturated 2-φ / superheated vapour.
  - Compute Re, friction factor:
    - Single-phase: Colebrook or Haaland on Re_SP.
    - Two-phase: liquid-only Δp × Müller-Steinhagen-Heck multiplier Φ²_LO (default), with Friedel and Lockhart-Martinelli selectable.
  - Compute h_inner:
    - Subcooled liquid: Dittus-Boelter or Gnielinski on liquid properties.
    - Two-phase: Chen or Kandlikar (saturated flow boiling); switch to post-CHF (Groeneveld) above critical quality x_crit estimated from Kim & Mudawar or similar.
    - Superheated vapour: Dittus-Boelter on vapour properties.
  - Compute Q̇_segment from full resistance network: 1/h_inner + ln(r_outer_pipe/r_inner_pipe)/(2πk_SS) + ln(r_outer_foam/r_inner_foam)/(2πk_foam(T̄)) + ln(r_outer_jacket/r_inner_jacket)/(2πk_Al) + 1/h_outer. Use cold-side T from current segment state, hot-side T from ambient profile.
  - Update enthalpy: h_{i+1} = h_i + (Q̇_segment + q_op_i)/ṁ.
  - Update pressure: p_{i+1} = p_i − (Δp_friction + Δp_acceleration + Δp_gravity)_segment.
    - Δp_acceleration = G²·(v_{i+1} − v_i) where v = 1/ρ.
    - Δp_gravity = ρ̄·g·Δz·sin(θ) where θ is the segment inclination.
  - Check choking: M = u / a_TP. If M ≥ 1, stop and flag the solver.
- Track and return arrays of p(z), T(z), h(z), x(z), ρ(z), u(z), regime(z), Q̇(z), M(z).

### 2.3 Outer solver
Two nested iterations (or one 2-D root-find on (D, t_foam)):
- D: chosen to satisfy Δp_total = Δp_max at the worst-case ambient.
- t_foam: chosen to satisfy x(z_target) ≥ 1 and T_exit ≥ T_exit_min.

Use `scipy.optimize.brentq` or a simple bisection with bounds D ∈ [4, 50] mm, t_foam ∈ [5, 200] mm.

### 2.4 Validation against the report's numbers
With Brewer's assumptions forced (constant ρ = 71 kg/m³, f = 0.013, no two-phase, T_amb = 295 K, k = 0.017 W/m·K, no aging, no supports, no radiation, latent only), the code must reproduce D = 7.52 mm and t = 83.1 mm and m/L = 1.801 kg/m to within 5 %. Add a `--brewer-reference` flag that locks those assumptions and prints the comparison.

---

## 3. Two-phase flow — what the LH₂ code must handle

Repeating §2.2 in checklist form, because this is the part of the problem most likely to be done wrong:

1. **Three regimes per pipe.** Subcooled liquid, saturated two-phase, superheated vapour. Switch model per segment.
2. **Density swing ~55×** (LH₂ 71 → GH₂ at 20 K ≈ 1.3 kg/m³). Acceleration pressure drop is dominant in high-x segments; cannot be ignored as in single-phase liquid flow.
3. **Two-phase friction multiplier.** Müller-Steinhagen-Heck is the default for cryogenic horizontal flow. Lockhart-Martinelli and Friedel as alternatives.
4. **Flow regime map.** Compute regime per segment (Baker or Mandhane for horizontal, Hewitt-Roberts for vertical). Wavy flow dominates for LH₂ — NASA: "wavy flow dominates due to the extremely low liquid-to-gas density ratio of hydrogen."
5. **Boiling heat transfer.** Chen / Kandlikar pre-CHF, Groeneveld post-CHF. Predict x_crit (dryout).
6. **Choking check.** Two-phase speed of sound is much lower than either single-phase value — Wallis or Wood model for a_TP. Flag if M ≥ 1 anywhere.
7. **Saturation tracking.** T_sat = T_sat(p(z)); cold-side temperature for the heat-leak calc is segment-local, not a global 20 K.
8. **Design target shifts.** Instead of "boil-off ≤ 1 %", the design solves for the axial position where x = 1 and the exit superheat T_exit − T_sat(p_exit).
9. **Slip ratio.** Default to Homogeneous Equilibrium Model (S = 1). Optionally support Zivi or Chisholm slip models.
10. **Orientation.** Segments must carry an inclination angle θ for the gravity term and for regime-map selection.
11. **Stiff properties near saturation.** Use (p, h) state queries, never (p, T), inside the dome. Step in enthalpy.
12. **Numerical safety.** If CoolProp returns a phase-boundary error, fall back to bracketing the saturation curve manually with `PQ_INPUTS` and interpolating.

---

## 4. CcH₂ requirements (supercritical case)

CcH₂ is single-phase supercritical throughout, so most of §3 disappears. What changes from the LH₂ case:

1. **Operating range.** p ∈ [250, 350] bar (or user-specified), T ∈ [25, 110] K. Always above p_crit (13 bar) and the line never crosses into the two-phase dome.
2. **No two-phase modelling.** Single-phase compressible flow throughout. Δp_friction with Colebrook, Δp_acceleration from ρ(p,T), Δp_gravity from segment-local ρ.
3. **Structural sizing is the dominant pipe-wall driver.** Hoop stress σ_h = p·D/(2·t_ss) with safety factor SF ≥ 2.5. For 316L at 20 K, design allowable σ_allow ≈ 250 MPa (verify against ASME B31.12 hydrogen piping code). t_ss will end up several mm, not 0.5 mm.
4. **Material constraints.** 316L is acceptable for cryo-H₂ provided cold-work and hydrogen exposure are bounded. The Imperial / arXiv work on 316plus at 20 K shows ductility losses of 40–50 % under hydrogen exposure; the code should expose `sigma_allow` as a temperature- and exposure-dependent input rather than a constant.
5. **No latent heat term.** Boil-off is undefined; instead constrain T_exit and/or ρ_exit at the engine interface.
6. **Heat ingress sets exit temperature.** Same resistance network as §2.2, but cold-side T rises monotonically along the line; the design criterion is `T_exit ≤ T_exit_max` (e.g. 110 K to stay in the CcH₂ operating envelope) or `T_exit` within a target window for the downstream heat-exchanger.
7. **Density advantage.** CcH₂ at ~80 g/L vs LH₂ at ~71 g/L → smaller bore for the same ṁ. The solver will give a noticeably smaller D.
8. **No bellows required for liquid–vapour boundary, but thermal contraction over 32 m between 316L and Al-6061 still ≈ 50 mm; expansion provisions remain mandatory.**
9. **No vacuum jacket option carried forward**, same as LH₂ case, but flag in the report that VJ is the higher-performance alternative.
10. **Embrittlement is more aggressive than LH₂.** Atomic hydrogen "only originates from the gaseous component, not liquid hydrogen" — apply a tighter de-rating on σ_allow for CcH₂ than for LH₂.

---

## 5. Target code architecture

```
hycool_pipe/
├── __init__.py
├── fluids.py          # CoolProp wrappers, ortho-para term
├── geometry.py        # routing, segment generation, inclination
├── ambient.py         # T_amb(z, mission_phase), h_outer(altitude)
├── materials.py       # k_SS(T), k_Al(T), k_foam(T̄), sigma_allow(T, H2-exposure)
├── friction.py        # Colebrook, Haaland, MSH, Friedel, LM multipliers
├── boiling.py         # Chen, Kandlikar, Groeneveld, x_crit
├── regimes.py         # Baker / Mandhane / Hewitt-Roberts maps
├── network.py         # resistance network, parasitic supports, plumbing margin
├── solver.py          # outer (D, t) iteration; choking & constraint checks
├── cases/
│   ├── lh2.py         # subcritical config + Brewer reference mode
│   └── cch2.py        # supercritical config
├── tests/
│   ├── test_brewer_reference.py
│   ├── test_property_lookups.py
│   ├── test_two_phase_regimes.py
│   └── test_cch2_hoop_stress.py
└── cli.py             # `python -m hycool_pipe lh2 --m-dot 71e-3 --L 32 ...`
```

Pseudocode for the inner march:

```python
def march(D, t_foam, m_dot, L, n_seg, p_in, h_in, ambient, case):
    dz = L / n_seg
    p, h = p_in, h_in
    log = []
    for i in range(n_seg):
        T = CP.PropsSI("T", "P", p, "Hmass", h, case.fluid)
        rho = CP.PropsSI("D", "P", p, "Hmass", h, case.fluid)
        x = quality(p, h, case.fluid)              # -1 if subcooled, 2 if superheated
        regime = classify(p, h, x, case.fluid)
        f = friction(regime, Re(D, m_dot, mu(p, h)))
        h_in_coef = h_coef_inner(regime, p, h, m_dot, D, case.fluid)
        h_out_coef = h_coef_outer(ambient.altitude(i), ambient.T(i))
        R = resistance_network(D, t_ss, t_foam, t_al,
                               k_SS, k_foam(T_log_mean), k_Al,
                               h_in_coef, h_out_coef)
        Q_seg = (ambient.T(i) - T) / R * dz
        Q_seg *= plumbing_margin                    # 1.3-1.5
        h_next = h + (Q_seg + q_op(T, case)) / m_dot
        # pressure: friction + accel + gravity
        u = m_dot / (rho * pi * D**2 / 4)
        dp_f = phi2_LO(regime) * f * (dz/D) * 0.5 * rho * u**2
        rho_next = CP.PropsSI("D", "P", p - dp_f, "Hmass", h_next, case.fluid)
        u_next = m_dot / (rho_next * pi * D**2 / 4)
        dp_a = m_dot * (u_next - u) / (pi * D**2 / 4)
        dp_g = rho * g * dz * sin(ambient.theta(i))
        p_next = p - dp_f - dp_a - dp_g
        # choking
        a_TP = sound_speed_two_phase(p, h, case.fluid)
        if u >= a_TP: raise ChokedFlow(i)
        log.append((i*dz, p, T, h, rho, u, x, Q_seg, regime))
        p, h = p_next, h_next
    return Result(log, p_exit=p, T_exit=T_from(p, h), x_exit=quality(p, h))
```

Outer solver:

```python
def size(case, m_dot, L, constraints):
    def residual(D, t):
        res = march(D, t, m_dot, L, ...)
        return [res.dp_total - constraints.dp_max,
                res.x_at_target - constraints.x_target]
    D, t = fsolve(residual, x0=[0.01, 0.05], bounds=...)
    return D, t, mass_per_length(D, t)
```

---

## 6. Validation / verification checks

1. **Brewer reproduction.** With `--brewer-reference`, reproduce D = 7.52 mm, t = 83.1 mm, m/L = 1.801 kg/m to within 5 %.
2. **Property sanity.** At (p = 1 bar, T = 20 K), CoolProp parahydrogen should give ρ_L ≈ 70.8 kg/m³, h_fg ≈ 446 kJ/kg.
3. **Single-phase limit.** A 100 % superheated GH₂ run should converge to standard Darcy–Weisbach with no surprises.
4. **Two-phase choking sentinel.** Construct a deliberately undersized line (D small, ṁ high) and assert ChokedFlow is raised.
5. **CcH₂ structural sanity.** At p = 350 bar, D = 5 mm, the solver should report t_ss ≥ ~0.6 mm (σ_h·SF = σ_allow).
6. **Mass conservation.** ṁ in = ṁ out per segment.
7. **Energy conservation.** Σ Q̇_segment = ṁ·(h_exit − h_in) + Δ(½ṁu²) within numerical tolerance.
8. **Convergence study.** Halve dz, results should change <1 %.

---

## 7. CLI

```
python -m hycool_pipe lh2 \
  --m-dot 0.071 --L 32 --n-bends 10 \
  --p-in 2.0e5 --T-in 20 \
  --dp-max 1.0e5 --T-exit-min 250 \
  --ambient hot-ground \
  --fluid parahydrogen \
  --plumbing-margin 1.3 --foam-aging 1.3 \
  --output results_lh2.json --plot

python -m hycool_pipe cch2 \
  --m-dot 0.071 --L 32 --n-bends 10 \
  --p-in 350e5 --T-in 50 \
  --T-exit-max 110 \
  --ambient hot-ground \
  --fluid hydrogen --sigma-allow 250e6 --sf 2.5 \
  --output results_cch2.json --plot
```

Outputs (JSON): D, t_foam, t_ss, t_al, m/L, total mass, Δp_total, T_exit, x_exit, choked (bool), arrays of all per-segment quantities. Plots: p(z), T(z), x(z), Q̇(z), regime(z).

---

## 8. References to cite in code comments and report

- G. D. Brewer, *Hydrogen Aircraft Technology*, Routledge, 1991.
- Johnson, Baltman, Koci, "Assessment of Insulation Systems for Aircraft Liquid Hydrogen Tanks," NASA / CEC 2023.
- Lockhart & Martinelli (1949); Friedel (1979); Müller-Steinhagen & Heck (1986) — two-phase friction multipliers.
- Chen (1966), Kandlikar (1990) — saturated flow-boiling heat transfer.
- Groeneveld (1973) — post-CHF film boiling.
- Kim & Mudawar — CHF correlation for cryogenic flow.
- Tseng et al., "Thermal conductivity of polyurethane foams from room temperature to 20 K."
- NIST REFPROP / CoolProp parahydrogen EOS.
- ASME B31.12, *Hydrogen Piping and Pipelines*, for structural-allowable inputs.

---

## 9. Out of scope (note in code, do not implement)

- Vacuum jacket variant.
- MLI variant.
- Composite tank-wall material.
- Transient chilldown (steady-state cruise + hot-ground only).
- Active vapour recirculation / boil-off recovery.

These should be left as TODO hooks (e.g. `class VacuumJacketInsulation(InsulationBase): raise NotImplementedError`) so the architecture admits them later.

---

## 10. First-session task list for Claude Code

1. Set up the package layout in §5.
2. Port `pipe_python.py` into `cases/lh2.py` as a baseline, behind the `--brewer-reference` flag.
3. Wire CoolProp into `fluids.py`.
4. Implement Colebrook + Haaland in `friction.py`, single-phase only first.
5. Replace constants with CoolProp lookups; verify the Brewer reference still matches.
6. Add the segmented march in `solver.py`, single-phase only.
7. Add two-phase friction (MSH) and boiling h-coef (Chen).
8. Add regime classification.
9. Add CcH₂ case + hoop-stress structural sizing.
10. Add the validation tests in §6.
11. CLI + JSON output + plotting.

End of spec.
