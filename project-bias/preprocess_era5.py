"""Command line program for ERA5 data pre-processing."""

import glob
import argparse

import numpy as np
import xarray as xr
import xclim
import cmdline_provenance as cmdprov


var_to_cmor_name = {
    'mx2t': 'tasmax',
    'tp': 'pr',
    'latitude': 'lat',
    'longitude': 'lon',
}

cmor_var_attrs = {
    'tasmax': {
        'long_name': 'Daily Maximum Near-Surface Air Temperature',
        'standard_name': 'air_temperature',
    },
    'pr': {
        'long_name': 'Precipitation',
        'standard_name': 'precipitation_flux',
    },
    'lat': {
        'long_name': 'latitude',
        'standard_name': 'latitude',
        'axis': 'Y',
        'units': 'degrees_north',
        'bounds': 'lat_bnds'
    },
    'lon': {
        'long_name': 'longitude',
        'standard_name': 'longitude',
        'axis': 'X',
        'units': 'degrees_east',
        'bounds': 'lon_bnds'
    },
}

output_units = {
    'tasmax': 'degC',
    'pr': 'mm d-1',
}


def convert_units(da, target_units):
    """Convert units.

    Parameters
    ----------
    da : xarray DataArray
        Input array containing a units attribute
    target_units : str
        Units to convert to

    Returns
    -------
    da : xarray DataArray
       Array with converted units
    """

    with xr.set_options(keep_attrs=True):
        da = xclim.units.convert_units_to(da, target_units)
    
    if target_units == 'degC':
        da.attrs['units'] = 'degC'

    return da


def fix_metadata(ds, var):
    "Apply metadata fixes."""

    dims = list(ds[var].dims)
    dims.remove('time')
    units = ds[var].attrs['units']
    for varname in dims + [var]:
        if varname in var_to_cmor_name:
            cmor_var = var_to_cmor_name[var]
            ds = ds.rename({varname: cmor_var})
        else:
            cmor_var = varname
        ds[cmor_var].attrs = cmor_var_attrs[cmor_var]
    ds[cmor_var].attrs['units'] = units

    return ds


def main(args):
    """Run the program."""

    cmor_var = var_to_cmor_name[args.var]
    outdir = '/g/data/xv83/unseen-projects/outputs/bias/data/'
    new_log = cmdprov.new_log()
    
    for year in np.arange(1940, 2025):
        infiles = sorted(glob.glob(f'/g/data/rt52/era5/single-levels/reanalysis/{args.var}/2025/*.nc'))
        if not infiles:
            raise OSError(f'No input files for variable {args.var} and year {year}')

        input_ds = xr.open_mfdataset(infiles)
        if args.var == 'mx2t':
            input_ds = input_ds.resample(time='D').max('time', keep_attrs=True)
        elif args.var == 'tp':
            input_ds = input_ds.resample(time='D').sum('time', keep_attrs=True)
            assert input_ds[args.var].attrs['units'] == 'm'
            input_ds[args.var].attrs['units'] = 'm d-1'
        else:
            raise ValueError(f'Unsupported variable {args.var}')

        output_ds[args.var] = convert_units(output_ds[args.var], output_units[cmor_var])
        output_ds = fix_metadata(output_ds, args.var)
        output_ds.attrs['history'] = new_log

        outpath = f'{outdir}/{cmor_var}_ERA5_day_gn_{year}0101-{year}1231.nc'
        print(outpath)
        output_ds.to_netcdf(outpath)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )     
    parser.add_argument(
        "var",
        type=str,
        choices=('mx2t', 'tp'),
        help="input variable"
    )
    args = parser.parse_args()
    main(args)
