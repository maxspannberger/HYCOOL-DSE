from math import log, pi


def mass_per_meter(rho_material, r_inner, r_outer):
    return rho_material * pi * (r_outer ** 2 - r_inner ** 2)


def run_pipe_analysis(
    m_dot,
    length,
    bends,
):
    # Assumed values (fixed).
    rho = 71
    f = 0.013
    k_minor = 0.02
    t_lh2 = 20
    t_ambient = 295
    k_insulation = 0.0173
    h_lat = 446000

    # Fixed geometry for current design stage.
    selected_diameter = 0.00752
    selected_insulation = 0.0831

    # Material properties and layer thicknesses.
    t_ss = 0.0005
    t_al = 0.0005
    rho_ss = 8000
    rho_foam = 40
    rho_al = 2700

    # Pressure drop for fixed diameter.
    k_total = k_minor * bends
    p_diff_pa = (
        8 / (pi ** 2 * rho)
        * ((m_dot ** 2) / (selected_diameter ** 4))
        * (f * length / selected_diameter + k_total)
    )

    diameter_mm = selected_diameter * 1000
    p_diff_kpa = p_diff_pa / 1000

    # Heat leak and boil-off for fixed insulation thickness.
    r1 = (selected_diameter / 2) + t_ss
    r2 = r1 + selected_insulation
    r_cond = log(r2 / r1) / (2 * pi * k_insulation)
    q_dot = (t_ambient - t_lh2) / r_cond
    q_total = q_dot * length
    boil_off_percent_per_m = q_dot / (h_lat * m_dot / 100)
    boil_off_percent_total = q_total / (h_lat * m_dot / 100)

    # Mass per meter for selected geometry.
    r_i_ss = selected_diameter / 2
    r_o_ss = r_i_ss + t_ss
    r_i_foam = r_o_ss
    r_o_foam = r_i_foam + selected_insulation
    r_i_al = r_o_foam
    r_o_al = r_i_al + t_al

    m_ss = mass_per_meter(rho_ss, r_i_ss, r_o_ss)
    m_foam = mass_per_meter(rho_foam, r_i_foam, r_o_foam)
    m_al = mass_per_meter(rho_al, r_i_al, r_o_al)
    m_total = m_ss + m_foam + m_al

    return {
        "diameter_mm": diameter_mm,
        "pressure_drop_kpa": p_diff_kpa,
        "insulation_m": selected_insulation,
        "heat_input_per_m_w": q_dot,
        "total_heat_input_w": q_total,
        "boil_off_percent_per_m": boil_off_percent_per_m,
        "boil_off_percent_total": boil_off_percent_total,
        "m_ss": m_ss,
        "m_foam": m_foam,
        "m_al": m_al,
        "m_total": m_total,
        "r_i_ss": r_i_ss,
        "r_o_ss": r_o_ss,
        "r_o_foam": r_o_foam,
        "r_o_al": r_o_al,
    }