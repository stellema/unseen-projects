"""Utility functions."""

import glob

import matplotlib.pyplot as plt
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
from unseen import moments


lat = {}
lon = {}

lat['Katherine'] = -14.5
lon['Katherine'] = 132.3

lat['Surat'] = -27.2
lon['Surat'] = 149.1

lat['Miena'] = -42.0
lon['Miena'] = 146.7


def get_obs_data(metric, location):
    """Get obs data"""

    var = {'txx': 'tasmax', 'rx1day': 'pr'}
    obs_file = glob.glob(f'/g/data/xv83/unseen-projects/outputs/bias/data/{metric}_AGCD-CSIRO_*_AUS300i.nc')[0]
    ds_obs = fileio.open_dataset(obs_file)
    da_obs = ds_obs[var[metric]].sel({'lat': lat[location], 'lon': lon[location]}, method='nearest')
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
    da_model = ds_model[var[metric]].sel({'lat': lat[location], 'lon': lon[location]}, method='nearest')
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
        time_period=['1972-01-01', '2018-12-31']
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


def quantile_correction(da_model_detrended_stacked, da_obs_detrended, metric):
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
    gev_xvals = np.arange(-20, 300, 0.5)

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
    plt.title(f'{metric} (Jul-Jun) at {location}')

    xmax = np.max([da_model_detrended_stacked.values.max(), da_obs_detrended.values.max()]) + 1
    xmin = np.min([da_model_detrended_stacked.values.min(), da_obs_detrended.values.min()]) - 1
    plt.xlim(xmin, xmax)
    plt.legend(fontsize='small')
    #plt.savefig(f'rx1day_{location}_gevs.png', bbox_inches='tight', facecolor='white')
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

