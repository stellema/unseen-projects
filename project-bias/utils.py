"""Utility functions."""

import glob
import logging

import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs
import pandas as pd
import numpy as np
import xarray as xr
from scipy.stats import genextreme as gev
from scipy import stats
import xclim as xc
from xclim import sdba
from xclim.sdba import nbutils

from unseen import fileio
from unseen import eva
from unseen import bias_correction
from unseen import time_utils
from unseen import similarity
from unseen import stability
from unseen import moments


lat = {}
lon = {}

lat['Katherine'] = -14.5
lon['Katherine'] = 132.3

lat['Surat'] = -27.2
lon['Surat'] = 149.1

lat['Miena'] = -42.0
lon['Miena'] = 146.7

lat_array = np.arange(-42, -11, 3)
lon_array = np.arange(113.5, 153, 3)


def get_obs_data(metric, location):
    """Get obs data"""

    var = {'txx': 'tasmax', 'rx1day': 'pr'}
    obs_file = glob.glob(f'/g/data/xv83/unseen-projects/outputs/bias/data/{metric}_AGCD-CSIRO_*_AUS300i.nc')[0]
    ds_obs = fileio.open_dataset(obs_file)
    if type(location) == str:
        da_obs = ds_obs[var[metric]].sel({'lat': lat[location], 'lon': lon[location]}, method='nearest')
    else:
        lat_index, lon_index = location
        da_obs = ds_obs[var[metric]].isel({'lat': lat_index, 'lon': lon_index})
    da_obs = da_obs.compute()

    return da_obs


def detrend_obs(da_obs):
    """Linearly detrend obs data."""

    linear_fit_obs = np.polyfit(da_obs.time.dt.year.values, da_obs.values, 1)
    linear_data_obs = np.polyval(linear_fit_obs, da_obs.time.dt.year.values)
    base_mean_obs = da_obs.sel(time=slice('1972-01-01', '2018-12-31')).mean().values
    da_obs_detrended = (da_obs - linear_data_obs) + base_mean_obs
    da_obs_detrended.attrs = da_obs.attrs

    return da_obs_detrended, linear_data_obs


def get_model_data(metric, model, location):
    """Get grid point data for a single metric/model combination"""

    var = {'txx': 'tasmax', 'rx1day': 'pr'}
    model_file = glob.glob(f'/g/data/xv83/unseen-projects/outputs/bias/data/{metric}_{model}-*_*_annual-jul-to-jun_AUS300i.nc')[0]
    ds_model = fileio.open_dataset(model_file)
    if type(location) == str:
        da_model = ds_model[var[metric]].sel({'lat': lat[location], 'lon': lon[location]}, method='nearest')
    else:
        lat_index, lon_index = location
        da_model = ds_model[var[metric]].isel({'lat': lat_index, 'lon': lon_index})
    da_model = da_model.compute()
    da_model_stacked = da_model.dropna('lead_time').stack({'sample': ['ensemble', 'init_date', 'lead_time']})

    return da_model_stacked


def detrend_model(da_model_stacked):
    """linearly detrend model data."""

    linear_fit_model = np.polyfit(da_model_stacked.time.dt.year.values, da_model_stacked.values, 1)
    linear_data_model = np.polyval(linear_fit_model, np.unique(da_model_stacked.time.dt.year.values))
    da_model_stacked_base = time_utils.select_time_period(da_model_stacked.copy(), ['1972-01-01', '2018-12-31'])
    base_mean_model = da_model_stacked_base.mean().values
    value_year_pairs = np.column_stack((da_model_stacked.values, da_model_stacked.time.dt.year.values))
    detrended_data = []
    for value, year in value_year_pairs:
        af = np.polyval(linear_fit_model, year)
        detrended_value = value - af + base_mean_model
        detrended_data.append(detrended_value)
    detrended_data = np.array(detrended_data)
    da_model_detrended_stacked = da_model_stacked * 0 + detrended_data
    da_model_detrended_stacked.attrs = da_model_stacked.attrs
    da_model_detrended = da_model_detrended_stacked.unstack()

    return da_model_detrended, da_model_detrended_stacked, linear_data_model


def mean_correction(da_model_detrended, da_obs_detrended, metric):
    """Apply a mean correction."""

    methods = {'txx': 'additive', 'rx1day': 'multiplicative'}
    method = methods[metric]  
    bias = bias_correction.get_bias(
        da_model_detrended,
        da_obs_detrended,
        method,
        time_rounding='A',
    )
    da_model_detrended_bc = bias_correction.remove_bias(da_model_detrended, bias, method)
    da_model_detrended_bc = da_model_detrended_bc.compute()
    da_model_detrended_bc_stacked = da_model_detrended_bc.dropna('lead_time').stack(
        {'sample': ['ensemble', 'init_date', 'lead_time']}
    )

    return da_model_detrended_bc_stacked


def _get_smooth_adjustment_factor(value, data, af_cubic_fit):
    """Find the adjustment factor."""

    quantile = stats.percentileofscore(data, value) / 100
    af = np.polyval(af_cubic_fit, quantile)

    return af


def quantile_correction(da_model_detrended_stacked, da_obs_detrended, metric, plot_af=False):
    """Apply a mean correction."""

    methods = {'txx': 'additive', 'rx1day': 'multiplicative'}
    method = methods[metric]

    nquantiles = 10
    quantile_array = xc.sdba.utils.equally_spaced_nodes(nquantiles)
    da_model_detrended_q = nbutils.quantile(
        da_model_detrended_stacked,
        quantile_array,
        ['sample']
    )
    da_obs_detrended_q = nbutils.quantile(
        da_obs_detrended,
        quantile_array,
        ['time']
    )

    if method == 'additive':
        bias = da_model_detrended_q.values - da_obs_detrended_q.values
    elif method == 'multiplicative':
        bias = da_model_detrended_q.values / da_obs_detrended_q.values
    bias_cubic_fit = np.polyfit(quantile_array, bias, 3)
    
    vget_smooth_adjustment_factor = np.vectorize(
        _get_smooth_adjustment_factor,
        excluded=['data', 'af_cubic_fit']
    )
    af = vget_smooth_adjustment_factor(
        da_model_detrended_stacked.values,
        data=da_model_detrended_stacked.values,
        af_cubic_fit=bias_cubic_fit,
    )
    if method == 'additive':
        da_model_detrended_stacked_bc = da_model_detrended_stacked - af
    elif method == 'multiplicative':
        da_model_detrended_stacked_bc = da_model_detrended_stacked / af

    if plot_af:
        plt.plot(quantile_array, bias, marker='o')
        cubic_data = np.polyval(bias_cubic_fit, quantile_array)
        plt.plot(quantile_array, cubic_data, marker='o')
        plt.xlabel('quantile')
        plt.ylabel(f'{method} adjustment factor')
        plt.show()

    return da_model_detrended_stacked_bc


def plot_model_data(da_model_stacked, da_model_detrended_stacked, linear_data_model, metric):
    """Plot raw and detrended model data"""

    ylabel = {'txx': 'temperature (C)', 'rx1day': 'rainfall (mm)'}
    fig, ax = plt.subplots(figsize=[9, 5])
    years = da_model_stacked['time'].dt.year.values
    unique_years = np.unique(da_model_stacked.time.dt.year.values)
    plt.scatter(
        years,
        da_model_stacked.values,
        marker='o',
        color='tab:blue',
        alpha=0.4,
        label='raw data'
    )
    plt.plot(unique_years, linear_data_model, color='tab:cyan')
    plt.scatter(
        years,
        da_model_detrended_stacked.values,
        marker='o',
        color='tab:orange',
        alpha=0.4,
        label='detrended data'
    )
    plt.xlim(unique_years[0] - 0.5, unique_years[-1] + 0.5)
    plt.title(metric)
    plt.ylabel(ylabel[metric])
    plt.xlabel('year')
    plt.legend()
    plt.grid()


def plot_distributions(
    metric,
    location,
    da_obs_detrended,
    gev_obs_detrended,
    da_model_detrended_stacked,
    gev_model_detrended,
    da_model_detrended_bc_mean_stacked,
    gev_model_detrended_bc_mean,
    da_model_detrended_bc_quantile_stacked,
    gev_model_detrended_bc_quantile,
):
    """Plot obs and model distributions."""

    fig, ax = plt.subplots(figsize=[8, 5])
    gev_xvals = np.arange(-20, 300, 0.1)

    da_model_detrended_stacked.plot.hist(bins=40, density=True, color='tab:blue', alpha=0.5)
    shape, loc, scale = gev_model_detrended
    pdf = gev.pdf(gev_xvals, shape, loc, scale)
    plt.plot(gev_xvals, pdf, color='tab:blue', linewidth=4.0, label='model (raw)')

    da_model_detrended_bc_mean_stacked.plot.hist(bins=40, density=True, color='tab:orange', alpha=0.5)
    shape, loc, scale = gev_model_detrended_bc_mean
    pdf = gev.pdf(gev_xvals, shape, loc, scale)
    plt.plot(gev_xvals, pdf, color='tab:orange', linewidth=4.0, label='model (mean correction)')

    da_model_detrended_bc_mean_stacked.plot.hist(bins=40, density=True, color='tab:green', alpha=0.5)
    shape, loc, scale = gev_model_detrended_bc_quantile
    pdf = gev.pdf(gev_xvals, shape, loc, scale)
    plt.plot(gev_xvals, pdf, color='tab:green', linewidth=4.0, label='model (quantile correction)')

    da_obs_detrended.plot.hist(bins=40, density=True, color='tab:gray', alpha=0.7)
    shape, loc, scale = gev_obs_detrended
    pdf = gev.pdf(gev_xvals, shape, loc, scale)
    plt.plot(gev_xvals, pdf, color='tab:gray', linewidth=4.0, label='obs (AGCD)')

    plt.xlabel(metric)
    plt.ylabel('probability')
    if type(location) == str:
        plt.title(f'{metric} (Jul-Jun) at {location}')
    else:
        lat_index, lon_index = location
        lat = float(da_obs_detrended.lat)
        lon = float(da_obs_detrended.lon)
        plt.title(f'{metric} (Jul-Jun) at lat {lat:.1f} and lon {lon:.1f}')

    xmax = np.max([da_model_detrended_stacked.values.max(), da_obs_detrended.values.max()]) + 1
    xmin = np.min([da_model_detrended_stacked.values.min(), da_obs_detrended.values.min()]) - 1
    plt.xlim(xmin, xmax)
    plt.legend(fontsize='small')
    #plt.savefig(f'rx1day_{location}_gevs.png', bbox_inches='tight', facecolor='white')
    plt.show()


def plot_return_curves(
    metric,
    location,
    da_obs_detrended,
    gev_obs_detrended,
    da_model_detrended,
    gev_model_detrended,
    da_model_detrended_bc_mean,
    gev_model_detrended_bc_mean,
    da_model_detrended_bc_quantile,
    gev_model_detrended_bc_quantile,
):
    """Plot return curves"""

    fig = plt.figure(figsize=[7, 6])
    ax = fig.add_subplot(111)

    return_periods_model_detrended, return_values_model_detrended = stability.return_curve(
        da_model_detrended, 'gev', params=gev_model_detrended,
    )
    ax.plot(return_periods_model_detrended, return_values_model_detrended, label='model (raw)', color='tab:blue')

    return_periods_model_detrended_bc_mean, return_values_model_detrended_bc_mean = stability.return_curve(
        da_model_detrended_bc_mean, 'gev', params=gev_model_detrended_bc_mean,
    )
    ax.plot(
        return_periods_model_detrended_bc_mean,
        return_values_model_detrended_bc_mean,
        label='model (mean correction)',
        color='tab:orange'
    )

    return_periods_model_detrended_bc_quantile, return_values_model_detrended_bc_quantile = stability.return_curve(
        da_model_detrended_bc_quantile, 'gev', params=gev_model_detrended_bc_quantile,
    )
    ax.plot(
        return_periods_model_detrended_bc_quantile,
        return_values_model_detrended_bc_quantile,
        label='model (quantile correction)',
        color='tab:green'
    )

    return_periods_obs_detrended, return_values_obs_detrended = stability.return_curve(
        da_obs_detrended, 'gev', params=gev_obs_detrended,
    )
    ax.plot(return_periods_obs_detrended, return_values_obs_detrended, label='AGCD', color='black', linewidth=2.0)

    ax.legend()
    ax.set_xscale('log')
    ax.set_xlabel('return period (years)')
    ax.set_ylabel(metric)
    ax.set_title('Return periods corresponding to GEV fits')
    ymin, ymax = ax.get_ylim()
    if ymin < 0:
        ymin = 0
    ax.set_ylim([ymin, ymax])
    ax.grid(which='both')
    plt.show()


def fidelity_tests(da_model_detrended, da_obs_detrended, da_model_detrended_bc):
    """Perform fidelity tests on target model-based data."""
    
    similarity_scores = similarity.similarity_tests(da_model_detrended_bc, da_obs_detrended)
    print('KS score:', similarity_scores['ks_statistic'].values)
    print('KS p-value:', similarity_scores['ks_pval'].values)
    print('AD score:', similarity_scores['ad_statistic'].values)
    print('AD p-value:', similarity_scores['ad_pval'].values)

    moments.create_plot(
        da_model_detrended,
        da_obs_detrended,
        da_bc_fcst=da_model_detrended_bc,
    )


def get_gev_uncertainty(da_model, reference_return_values, name):
    """Get GEV uncertainty."""

    bootstrap_samples_dict = {}
    rng = np.random.default_rng(seed=0)
    n_bootstraps = 100
    for i in range(n_bootstraps):
        boot_data = rng.choice(da_model.values, size=da_model.shape, replace=True)
        gev_params = list(eva.fit_gev(boot_data))
        return_periods, return_values = stability.return_curve(boot_data, 'gev', params=gev_params)
        diff = return_values - reference_return_values
        bootstrap_samples_dict[i] = np.abs(diff)
    df = pd.DataFrame(bootstrap_samples_dict)
    df.index = return_periods
    ds = df.var(axis=1)
    ds.name = name

    return ds
    

def get_return_values(metric, location, model_dict, similarity_check=False):
    """Get return values for each dataset."""
    
    return_values_dict = {}
    gev_spread_dict = {}

    da_obs = get_obs_data(metric, location)
    da_obs_detrended, linear_data_obs = detrend_obs(da_obs)
    gev_obs_detrended = list(eva.fit_gev(da_obs_detrended.values))
    return_periods, return_values_obs = stability.return_curve(
        da_obs_detrended, 'gev', params=gev_obs_detrended,
    )
    return_values_dict[('obs', 'AGCD')] = return_values_obs
    gev_spread_obs = get_gev_uncertainty(
        da_obs_detrended,
        return_values_obs,
        name=('obs', 'AGCD'),
    )
    gev_spread_dict[('obs', 'AGCD')] = gev_spread_obs

    for model in model_dict:
        logging.info(f'start: {model}')
        da_model_stacked = get_model_data(metric, model, location)
        da_model_detrended, da_model_detrended_stacked, linear_data_model = detrend_model(da_model_stacked)
        da_model_detrended_stacked_bc_mean = mean_correction(da_model_detrended, da_obs_detrended, metric)
        if similarity_check:
            similarity_scores = similarity.similarity_tests(da_model_detrended_stacked_bc_mean.unstack(), da_obs_detrended)
            if float(similarity_scores['ks_pval'].values) < 0.05:
                continue
        da_model_detrended_stacked_bc_quantile = quantile_correction(da_model_detrended_stacked, da_obs_detrended, metric)
        gev_model_detrended = list(eva.fit_gev(da_model_detrended_stacked.values))
        gev_model_detrended_bc_mean = list(eva.fit_gev(da_model_detrended_stacked_bc_mean.values))
        gev_model_detrended_bc_quantile = list(eva.fit_gev(da_model_detrended_stacked_bc_quantile.values))
        return_periods, return_values_model_raw = stability.return_curve(
            da_model_detrended_stacked,
            'gev',
            params=gev_model_detrended,
        )
        gev_spread_model_raw = get_gev_uncertainty(
            da_model_detrended_stacked,
            return_values_model_raw,
            name=('model-raw', model),
        )
        return_periods, return_values_model_bc_mean = stability.return_curve(
            da_model_detrended_stacked_bc_mean,
            'gev',
            params=gev_model_detrended_bc_mean,
        )
        gev_spread_model_bc_mean = get_gev_uncertainty(
            da_model_detrended_stacked_bc_mean,
            return_values_model_bc_mean,
            name=('model-bc-mean', model),
        )
        return_periods, return_values_model_bc_quantile = stability.return_curve(
            da_model_detrended_stacked_bc_quantile,
            'gev',
            params=gev_model_detrended_bc_quantile,
        )
        gev_spread_model_bc_quantile = get_gev_uncertainty(
            da_model_detrended_stacked_bc_quantile,
            return_values_model_bc_quantile,
            name=('model-bc-quantile', model),
        )
        return_values_dict[('model-raw', model)] = return_values_model_raw
        return_values_dict[('model-bc-mean', model)] = return_values_model_bc_mean
        return_values_dict[('model-bc-quantile', model)] = return_values_model_bc_quantile
        gev_spread_dict[('model-raw', model)] = gev_spread_model_raw
        gev_spread_dict[('model-bc-mean', model)] = gev_spread_model_bc_mean
        gev_spread_dict[('model-bc-quantile', model)] = gev_spread_model_bc_quantile
        logging.info(f'end: {model}')

    return_values_df = pd.DataFrame(return_values_dict)
    return_values_df.index = return_periods
    return_values_df = return_values_df.drop([1.0])
    gev_spread_df = pd.DataFrame(gev_spread_dict)
    gev_spread_df.index = return_periods
    gev_spread_df = gev_spread_df.drop([1.0])
    
    return return_values_df, gev_spread_df


def uncertainty_breakdown(return_df, gev_spread_df):
    """Return curve uncertainty breakdown."""

    gev_spread = gev_spread_df.filter(like='model-bc-mean').mean(axis=1)
    G2 = gev_spread
    G = np.sqrt(G2)

    model_bc_mean_spread = return_df.filter(like='model-bc-mean').var(axis=1)
    M2 = model_bc_mean_spread
    M = np.sqrt(M2)
    
    B2_models = []
    for bias_method, model in return_df.filter(like='model-bc-mean').columns.values:
        B2 = return_df[[('model-bc-mean', model), ('model-bc-quantile', model)]].var(axis=1)
        B2.name = model
        B2_models.append(B2)
    B2_ensemble = pd.concat(B2_models, axis=1)
    bias_spread = B2_ensemble.mean(axis=1)
    B2 = bias_spread
    B = np.sqrt(B2)

    T2 = G2 + M2 + B2
    T = np.sqrt(T2)
    F = (G + M + B) / T

    ave_model_bc_mean = return_df.filter(like='model-bc-mean').mean(axis=1)

    obs = return_df[('obs', 'AGCD')]
    gev_spread_obs = gev_spread_df[('obs', 'AGCD')]
    O2 = gev_spread_obs
    O = np.sqrt(O2)

    uncertainty = [G, M, B, T, O]

    return obs, ave_model_bc_mean, uncertainty


def highlight_grid_box(ax, target_lat_index, target_lon_index, color='tab:red'):
    """Draw a red box around a grid point."""
    
    target_lat = lat_array[target_lat_index]
    target_lon = lon_array[target_lon_index]
    ax.plot(
        [target_lon - 1.5, target_lon + 1.5],
        [target_lat - 1.5, target_lat - 1.5],
        transform=ccrs.PlateCarree(),
        color=color,
        lw=2.5
    )
    ax.plot(
        [target_lon - 1.5, target_lon + 1.5],
        [target_lat + 1.5, target_lat + 1.5],
        transform=ccrs.PlateCarree(),
        color=color,
        lw=2.5
    )
    ax.plot(
        [target_lon - 1.5, target_lon - 1.5],
        [target_lat - 1.5, target_lat + 1.5],
        transform=ccrs.PlateCarree(),
        color=color,
        lw=2.5
    )
    ax.plot(
        [target_lon + 1.5, target_lon + 1.5],
        [target_lat - 1.5, target_lat + 1.5],
        transform=ccrs.PlateCarree(),
        color=color,
        lw=2.5
    )


def plot_grid_box(lat_index, lon_index):
    """Highlight a grid box on a map."""
    
    min_lat = lat_array.min() - 1.5
    max_lat = lat_array.max() + 1.5
    min_lon = lon_array.min() - 1.5
    max_lon = lon_array.max() + 1.5
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.add_feature(cartopy.feature.STATES)
    for lon in lon_array:
        ax.plot(
            [lon - 1.5, lon - 1.5],
            [min_lat, max_lat],
            transform=ccrs.PlateCarree(),
            color='0.5',
            lw=0.5
        )
    for lat in lat_array:
        ax.plot(
            [min_lon, max_lon],
            [lat - 1.5, lat - 1.5],
            transform=ccrs.PlateCarree(),
            color='0.5',
            lw=0.5
        )
    highlight_grid_box(ax, lat_array[lat_index], lon_array[lon_index])
    ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())
    plt.show()