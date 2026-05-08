import xarray as xr

nc_path = r'Climate Impact\fd16b8ba74e3acb261aa83a75e2d74f4.nc'

ds = xr.open_dataset(nc_path)

olr = ds['avg_tnlwrf'].sel(
    latitude=51.0,
    longitude=10.0,
    method='nearest'
).mean().values

print(f"Annual mean OLR at 51N, 10E: {olr:.2f} W/m2")