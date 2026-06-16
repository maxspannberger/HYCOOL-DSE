import xarray as xr
# needs install xarray, h5netcdf, netcdf4

from pathlib import Path
import os
import sys

root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

local_dir = Path(__file__).resolve().parent
nc_path = os.path.join(local_dir, "fd16b8ba74e3acb261aa83a75e2d74f4.nc")

ds = xr.open_dataset(nc_path)

olr = ds['avg_tnlwrf'].sel(
    latitude=51.0,
    longitude=10.0,
    method='nearest'
).mean().values

print(f"Annual mean OLR at 51N, 10E: {olr:.2f} W/m2")