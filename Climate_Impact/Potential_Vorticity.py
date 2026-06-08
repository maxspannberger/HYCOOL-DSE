import xarray as xr
# needs install xarray, h5netcdf, netcdf4

from pathlib import Path
import os
import sys

root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

local_dir = Path(__file__).resolve().parent
nc_path = os.path.join(local_dir, "d0403ac88bdf8461c770a9135180d3dd.nc")

ds = xr.open_dataset(nc_path, engine='netcdf4')
#print(ds)

#print(ds['pv'].pressure_level.values)

pv_500 = ds['pv'].sel(
    latitude=51.0,
    longitude=10.0,
    pressure_level=500.0
).mean().values * 1e6

pv_450 = ds['pv'].sel(
    latitude=51.0,
    longitude=10.0,
    pressure_level=450.0
).mean().values * 1e6

# Linear interpolation to 472 hPa
# (500 - 472) / (500 - 450) = 0.56 of the way from 500 to 450
weight = (500 - 472) / (500 - 450)
pv_472 = pv_500 + weight * (pv_450 - pv_500)
#print(f"Interpolated PV at 472 hPa (FL200): {pv_472:.4f} PVU")