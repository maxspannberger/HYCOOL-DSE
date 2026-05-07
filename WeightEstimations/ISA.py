

def isa(altitude_m: float) -> tuple[float, float, float]:
    """
    International Standard Atmosphere.
    Returns (T [K], p [Pa], rho [kg/m^3]) at given geopotential altitude.
    Valid for troposphere (h < 11 000 m) and lower stratosphere (h < 20 000 m).
    """
    T0, p0, rho0 = 288.15, 101_325.0, 1.225
    L, R, g = 0.0065, 287.05, 9.80665

    if altitude_m <= 11_000:
        T   = T0 - L * altitude_m
        p   = p0 * (T / T0) ** (g / (R * L))
        rho = p / (R * T)
    else:
        T11   = T0 - L * 11_000
        p11   = p0 * (T11 / T0) ** (g / (R * L))
        rho11 = p11 / (R * T11)
        T     = T11
        p     = p11 * np.exp(-g * (altitude_m - 11_000) / (R * T11))
        rho   = p / (R * T)

    return T, p, rho