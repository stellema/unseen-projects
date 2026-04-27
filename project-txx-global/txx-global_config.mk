# Configuration file for UNSEEN project on TXx

PROJECT_NAME=txx-global
ENV_DIR=/g/data/xv83/as3189/conda/envs/unseen
PROJECT_DIR=/g/data/xv83/unseen-projects/outputs/${PROJECT_NAME}
NOTEBOOK_IN_DIR=/g/data/xv83/unseen-projects/code
NOTEBOOK_OUT_DIR=/g/data/xv83/unseen-projects/code/project-${PROJECT_NAME}

## Metric calculation
VAR=tasmax
UNITS=degC
TIME_FREQ=YE-DEC
METRIC_OPTIONS=--variables ${VAR} --units ${VAR}='${UNITS}' --lat_bnds -60 90 --time_agg_dates --reset_times --verbose
# Limit the number of lead times to 9 (fix for DCPP EC-Earth3 members with 11 years of data)
METRIC_OPTIONS_FCST=--time_freq ${TIME_FREQ} --time_agg max --input_freq D --time_agg_min_tsteps 360 --output_chunks lead_time=50 --lead_dim_max_size 9
REFERENCE_TIME_PERIOD=1961-01-01 2018-12-30

## Labels
METRIC=txx
REGION=gn
TIMESCALE=annual-jan-to-dec
SIMILARITY_TEST=ks

# Shapefile for GEV parameters (mask invalid ocean points to avoid errors)
SHAPE_OVERLAP=0.1
SHAPEFILE=${PROJECT_DIR}/shapefiles/ne_110m_land_dissolved.shp

## Independence test options
INDEPENDENCE_OPTIONS=--confidence_interval 0.99 --n_resamples 1000
## Minimum lead time file options (independence file kwargs e.g., median min_lead over shapefile)
MIN_IND_LEAD_OPTIONS=--min_lead_kwargs variables=min_lead shapefile=${SHAPEFILE} shape_overlap=${SHAPE_OVERLAP} spatial_agg=${MIN_LEAD_SHAPE_SPATIAL_AGG}

## GEV distribution options
FITSTART=scipy_fitstart
GEV_TEST=bic
GEV_STATIONARY_OPTIONS=--fitstart ${FITSTART} --use_basinhopping --file_kwargs variables=${VAR} shapefile=${GEV_SHAPEFILE} shape_overlap=${SHAPE_OVERLAP}
GEV_NONSTATIONARY_OPTIONS=--covariate "time.year" --fitstart ${FITSTART} --use_basinhopping --file_kwargs variables=${VAR} shapefile=${GEV_SHAPEFILE} shape_overlap=${SHAPE_OVERLAP}
GEV_OBS_OPTIONS=--reference_time_period ${REFERENCE_TIME_PERIOD}

## Notebook options
MIN_LEAD_SHAPE_SPATIAL_AGG=median
TIME_AGG=maximum
# Non-stationary GEV covariate used for return levels
COVARIATE_BASE=2018
# Period for trend calculation (string converted to python)
GEV_TREND_PERIOD='[1961, 2018]'
# Dictionary of plot options for spatial analysis notebook (string converted to python)
PLOT_DICT='dict(metric="TXx", var="${VAR}", var_name="Temperature", units="°C", units_label="Temperature [°C]", freq="${TIME_FREQ}", cmap=cmap_dict["ipcc_temp_seq"].reversed(), cmap_anom=plt.cm.RdBu_r, ticks=np.arange(15, 60 + 5, 5), ticks_anom=np.arange(-8.5, 9.5, 1), ticks_anom_std=np.arange(-8.5, 9.5, 1), ticks_anom_pct=np.arange(-29, 31, 2), ticks_anom_ratio=np.arange(0.675, 1.325, 0.05), ticks_trend=np.around(np.arange(-0.85, 0.95, 0.1), 2), ticks_param_trend={"location": np.arange(-0.6, 0.61, 0.1), "scale": np.arange(-0.12, 0.122, 0.02)}, cbar_extend="both", acs_map_plot_kwargs=dict(name=None, mask_not_australia=False, agcd_mask=False, figsize=[12, 5], xlim=(-180, 180), ylim=(-60, 82), projection=ccrs.PlateCarree(central_longitude=0), coastlines=True, mask_ocean=True), map_plot_kwargs=dict(region=None, cbar_kwargs=dict(fraction=0.05), mask_not_australia=False, xlim=(-180, 180), ylim=(-60, 82), xticks=np.arange(-160, 180, 20), yticks=np.arange(-40, 80, 10), mask_ocean=True, coastlines=True))'

#  Plot additive/multiplicative bias corrected metric in spatial analysis notebook (True/False)#
PLOT_ADDITIVE_BC=1
PLOT_MULTIPLICATIVE_BC=0
