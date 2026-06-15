# Two-phase flow regime classification.
# Phase state: quality-based (subcooled / two-phase / superheated).
# Flow pattern: Mandhane et al. (1974) for horizontal; Hewitt & Roberts (1969) for vertical.
# NASA note: wavy/stratified flow dominates for LH2 due to extremely low rho_G/rho_L (~1/55).


class FlowRegime:
    SUBCOOLED_LIQUID    = "subcooled_liquid"
    SATURATED_TWO_PHASE = "saturated_two_phase"
    SUPERHEATED_VAPOUR  = "superheated_vapour"


# Flow pattern labels
class FlowPattern:
    BUBBLY      = "bubbly"
    SLUG        = "slug"
    WAVY        = "wavy"
    STRATIFIED  = "stratified"
    ANNULAR     = "annular"
    DISPERSED   = "dispersed"
    SINGLE_PHASE = "single_phase"


def classify_regime(x: float) -> str:
    """Return FlowRegime string for vapour quality x.

    x < 0  → subcooled liquid
    0≤x≤1  → saturated two-phase
    x > 1  → superheated vapour
    """
    if x < 0.0:
        return FlowRegime.SUBCOOLED_LIQUID
    if x > 1.0:
        return FlowRegime.SUPERHEATED_VAPOUR
    return FlowRegime.SATURATED_TWO_PHASE


def two_phase_pattern(
    G: float,
    x: float,
    rho_l: float,
    rho_g: float,
    D: float,
    orientation: str = "horizontal",
) -> str:
    """Flow pattern from Mandhane et al. (1974) for horizontal pipes.

    Uses superficial velocities j_L and j_G as coordinates.
    Boundaries are for air–water; LH2's low rho_G/rho_L ratio (~1/55) shifts
    the map toward stratified/wavy relative to air-water — see NASA note above.

    For vertical pipes (Hewitt & Roberts, 1969): stub, returns NotImplementedError.

    Parameters
    ----------
    G        : mass flux [kg/m²s]
    x        : vapour quality [0, 1]
    rho_l/g  : saturated densities [kg/m³]
    D        : inner diameter [m]  (unused here but reserved for future corrections)
    """
    if orientation == "vertical":
        raise NotImplementedError("Hewitt–Roberts vertical map not yet implemented.")

    j_G = x * G / rho_g          # gas superficial velocity [m/s]
    j_L = (1.0 - x) * G / rho_l  # liquid superficial velocity [m/s]

    # Mandhane (1974) horizontal map — approximate boundary conditions
    # (originally calibrated for air-water at ~1 bar)
    if j_G > 10.0:
        return FlowPattern.ANNULAR        # high gas velocity → annular
    if j_G > 3.0 and j_L < 0.1:
        return FlowPattern.ANNULAR
    if j_L > 3.0 and j_G < 0.5:
        return FlowPattern.BUBBLY         # high liquid rate → bubbly
    if j_L > 0.5 and j_G < 3.0:
        return FlowPattern.SLUG           # moderate both → slug
    if j_L < 0.1 and j_G < 3.0:
        return FlowPattern.STRATIFIED     # low both → stratified
    # Default for LH2: wavy (low rho_G/rho_L means liquid pools at bottom)
    return FlowPattern.WAVY
