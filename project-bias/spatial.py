import warnings
warnings.filterwarnings('ignore')
import argparse
import logging
logging.basicConfig(level=logging.INFO)

import numpy as np
import geopandas as gpd
import regionmask
import xarray as xr

from unseen import fileio

import utils


model_dict = {
    'BCC-CSM2-MR': 'tab:blue',
    'CAFE': 'tab:orange',
    'CMCC-CM2-SR5': 'tab:green',
    'CanESM5': 'tab:red',
    'EC-Earth3': 'tab:purple',
    'IPSL-CM6A-LR': 'tab:brown',
    'MIROC6': 'tab:pink',
    'MPI-ESM1-2-HR': 'tab:grey',
    'MRI-ESM2-0': 'tab:olive',
    'NorCPM1': 'tab:cyan',
}


def extract_closest_row(df, return_period):
    """Extract the row from df closest to the return period."""

    differences = np.array(np.abs(df.index - return_period))
    index = np.argmin(differences)

    return df.iloc[index]


def get_mask():
    """Get ocean mask."""

    infile = '/g/data/xv83/unseen-projects/outputs/bias/data/rx1day_AGCD-CSIRO_1901-2024_annual-jul-to-jun_AUS300i.nc'
    ds = fileio.open_dataset(infile)
    overlap_threshold = 0.67
    shape_gpd = gpd.read_file('/g/data/ia39/aus-ref-clim-data-nci/shapefiles/data/australia/australia.shp')
    shape_rgm = regionmask.from_geopandas(
        shape_gpd,
        names="AUS_NAME21",
        abbrevs="AUS_CODE21",
        name="australia"
    )
    frac = shape_rgm.mask_3D_frac_approx(ds)
    mask = frac.sel(region=0) >= overlap_threshold

    return mask


def main(args):
    """Run the program."""

    mask = get_mask()

    return_periods = np.array([100, 1000])
    nlevels = len(return_periods)
    nlats, nlons = mask.shape
    G_array = np.zeros([nlevels, nlats, nlons])
    M_array = np.zeros([nlevels, nlats, nlons])
    B_array = np.zeros([nlevels, nlats, nlons])
    T_array = np.zeros([nlevels, nlats, nlons])
    O_array = np.zeros([nlevels, nlats, nlons])
    MMM_array = np.zeros([nlevels, nlats, nlons])
    obs_array = np.zeros([nlevels, nlats, nlons])
    nmodels_array = np.zeros([nlats, nlons])
    for lat_index in range(nlats):
        for lon_index in range(nlons):
            logging.info(f'lat index: {lat_index}, lon index: {lon_index}')

            include_point = mask.isel({'lat': lat_index, 'lon': lon_index})
            if include_point:
                return_df, gev_spread_df = utils.get_return_values(
                    args.metric,
                    [lat_index, lon_index],
                    model_dict,
                    similarity_check=True
                )
                nmodels = return_df.filter(like='model-bc-mean').shape[1]
            else:
                nmodels = np.nan
            
            if nmodels >= 1:
                obs, MMM, uncertainty = utils.uncertainty_breakdown(return_df, gev_spread_df)
                G, M, B, T, O = uncertainty
                for level_index, return_period in enumerate(return_periods):
                    Gs = extract_closest_row(G, return_period)
                    Ms = extract_closest_row(M, return_period)
                    Bs = extract_closest_row(B, return_period)
                    Ts = extract_closest_row(T, return_period)
                    Os = extract_closest_row(O, return_period)
                    MMMs = extract_closest_row(MMM, return_period)
                    obss = extract_closest_row(obs, return_period)
                    GMBs = Gs + Ms + Bs
                    Gs_pct = (Gs / GMBs) * 100
                    Ms_pct = (Ms / GMBs) * 100
                    Bs_pct = (Bs / GMBs) * 100
                    G_array[level_index, lat_index, lon_index] = Gs_pct
                    M_array[level_index, lat_index, lon_index] = Ms_pct
                    B_array[level_index, lat_index, lon_index] = Bs_pct
                    T_array[level_index, lat_index, lon_index] = Ts
                    O_array[level_index, lat_index, lon_index] = Os
                    MMM_array[level_index, lat_index, lon_index] = MMMs
                    obs_array[level_index, lat_index, lon_index] = obss
                    nmodels_array[lat_index, lon_index] = nmodels
            else:
                nmodels_array[lat_index, lon_index] = nmodels
                for level_index in range(nlevels):
                    G_array[level_index, lat_index, lon_index] = np.nan
                    M_array[level_index, lat_index, lon_index] = np.nan
                    B_array[level_index, lat_index, lon_index] = np.nan
                    T_array[level_index, lat_index, lon_index] = np.nan
                    O_array[level_index, lat_index, lon_index] = np.nan
                    MMM_array[level_index, lat_index, lon_index] = np.nan
                    obs_array[level_index, lat_index, lon_index] = np.nan
                    
    units_dict = {'txx': 'Celsius', 'rx1day': 'mm'}
    units = units_dict[args.metric]
    ds_out = xr.Dataset(
        data_vars={
            'G': (['lev', 'lat', 'lon'], G_array, {'long_name': 'GEV uncertainty fraction (model)', 'units': '%'}),
            'M': (['lev', 'lat', 'lon'], M_array, {'long_name': 'model uncertainty fraction (model)', 'units': '%'}),
            'B': (['lev', 'lat', 'lon'], B_array, {'long_name': 'bias correction uncertainty fraction (model)', 'units': '%'}),
            'T': (['lev', 'lat', 'lon'], T_array, {'long_name': 'total model uncertainty (standard deviation)', 'units': units}),
            'O': (['lev', 'lat', 'lon'], O_array, {'long_name': 'total observations uncertainty (standard deviation)', 'units': units}),
            'MMM': (['lev', 'lat', 'lon'], MMM_array, {'long_name': 'multi-model mean (mean correction)', 'units': units}),
            'obs': (['lev', 'lat', 'lon'], obs_array, {'long_name': 'observations', 'units': units}),
            'nmodels': (['lat', 'lon'], nmodels_array, {'long_name': 'number of models', 'units': ' '}),
        },
        coords={
            'lev': (['lev',], return_periods, {'standard_name': 'return_period', 'long_name': 'return period', 'units': 'years'}),
            'lat': (['lat',], mask.lat.values, {'standard_name': 'latitude', 'long_name': 'latitude', 'units': 'degrees_north', 'axis': 'Y'}),
            'lon': (['lon',], mask.lon.values, {'standard_name': 'longitude', 'long_name': 'longitude', 'units': 'degrees_east', 'axis': 'X'}),
        },
        attrs={
            'metric': args.metric,
        },
    )
    ds_out.to_netcdf(args.outfile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )     
    parser.add_argument("metric", type=str, choices=('txx', 'rx1day'), help="metric")
    parser.add_argument("outfile", type=str, help="output file name")
    args = parser.parse_args()
    main(args)
