import xarray as xr
# needs install xarray, h5netcdf, netcdf4

from pathlib import Path
import os
import sys

root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

local_dir = Path(__file__).resolve().parent
nc_path = os.path.join(local_dir, "ad5b26c67b230cdb570420b82980e47b.nc")

ds = xr.open_dataset(nc_path, engine='netcdf4')
#print(ds)

# Extract PV at both pressure levels at 51N, 10E
pv_400 = ds['pv'].sel(
    latitude=51.0,
    longitude=10.0,
    pressure_level=400.0
).mean().values * 1e6

pv_350 = ds['pv'].sel(
    latitude=51.0,
    longitude=10.0,
    pressure_level=350.0
).mean().values * 1e6

# print(f"PV at 400 hPa: {pv_400:.4f} PVU")
# print(f"PV at 350 hPa: {pv_350:.4f} PVU")

# Linear interpolation to 376 hPa
# (400 - 376) / (400 - 350) = 0.48 of the way from 400 to 350
weight = (400 - 376) / (400 - 350)
pv_376 = pv_400 + weight * (pv_350 - pv_400)
# print(f"Interpolated PV at 376 hPa: {pv_376:.4f} PVU")