# Geometry Functions for Tank Design

from dataclasses import dataclass
import numpy as np

@dataclass
class TankShape:
    a: float        # major radius [m]
    c: float        # minor radius [m]
    l_s: float      # shell length [m]
    phi: float
    lam: float

def solve_shape(phi: float, lam: float, V: float) -> TankShape:
    """Closed-form: (phi, lam, V) -> physical dimensions."""
    if not (0.0 < lam < 1.0):
        raise ValueError(f"lambda must be in (0, 1); got {lam}.")
    if phi <= 0.0:
        raise ValueError(f"phi must be positive; got {phi}.")
    if V <= 0.0:
        raise ValueError(f"V must be positive; got {V}.")

    shape_factor = (4.0 / 3.0) + 2.0 * lam / (1.0 - lam)
    c = (V / (np.pi * phi * shape_factor)) ** (1.0 / 3.0)
    a = phi * c
    l_s = 2.0 * c * lam / (1.0 - lam)
    return TankShape(a=a, c=c, l_s=l_s, phi=phi, lam=lam)

def surface_area(shape: TankShape) -> float:
    """Total external area [m^2] (shell + two caps)."""
    

def shell_principal_radii(shape: TankShape, theta: float) -> tuple[float, float]:
    """Principal radii of curvature on the elliptical cylinder at angle theta."""

def cap_principal_radii(shape: TankShape, eta: float) -> tuple[float, float]:
    """Principal radii on the half-ellipsoid cap at meridional position eta."""

def volume(shape: TankShape) -> float:
    """Forward check: should reproduce V from solve_shape."""