import numpy as np
from pathlib import Path

def airfoil_area(dat_file, chord=2.86):
    """
    Calculate the enclosed airfoil area from a .dat file.

    Parameters
    ----------
    dat_file : str
        Path to the airfoil .dat file.
    chord : float, optional
        Chord length [m]. Default is 1.0.

    Returns
    -------
    area_normalized : float
        Area coefficient A/c².
    area_actual : float
        Physical area [m²] per unit span.
    """

    coords = []

    with open(dat_file, "r") as f:
        for line in f:
            parts = line.strip().split()

            # Skip headers and malformed lines
            if len(parts) != 2:
                continue

            try:
                x = float(parts[0])
                y = float(parts[1])
                coords.append([x, y])
            except ValueError:
                continue

    coords = np.array(coords)

    if len(coords) < 3:
        raise ValueError("Not enough coordinate points found in file.")

    # Close polygon if necessary
    if not np.allclose(coords[0], coords[-1]):
        coords = np.vstack([coords, coords[0]])

    x = coords[:, 0]
    y = coords[:, 1]

    # Shoelace formula
    area_normalized = 0.5 * abs(
        np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])
    )

    area_actual = area_normalized * chord**2

    return area_normalized, area_actual


def airfoil_folder_area(data_dir, chord=2.86):
    """Compute airfoil areas for all .dat files in a folder."""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Airfoil data folder not found: {data_dir}")

    results = []
    for dat_file in sorted(data_dir.glob("*.dat")):
        area_normalized, area_actual = airfoil_area(dat_file, chord=chord)
        results.append({
            "file": dat_file.name,
            "path": str(dat_file),
            "area_normalized": area_normalized,
            "area_actual": area_actual,
        })

    return results


if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parent / "Airfoil_Data"
    results = airfoil_folder_area(data_dir, chord=2.86)

    for row in results:
        print(f"{row['file']}: A/c² = {row['area_normalized']:.4f}, A = {row['area_actual']:.4f} m²")