# Configuration file for UNSEEN project on RX1day

PROJECT_NAME=rx1day-global
ENV_DIR=/g/data/xv83/as3189/conda/envs/unseen
PROJECT_DIR=/g/data/xv83/unseen-projects/outputs/${PROJECT_NAME}
NOTEBOOK_IN_DIR=/g/data/xv83/unseen-projects/code
NOTEBOOK_OUT_DIR=/g/data/xv83/unseen-projects/code/project-${PROJECT_NAME}

## Metric calculation
VAR=pr
UNITS=mm day-1
TIME_FREQ=YE-DEC
METRIC_OPTIONS=--variables ${VAR} --time_freq ${TIME_FREQ} --units ${VAR}='${UNITS}' --time_agg max --input_freq D --time_agg_min_tsteps 360 --time_agg_dates --reset_times  --verbose
# Limit the number of lead times to 9 (fix for DCPP EC-Earth3 members with 11 years of data)
METRIC_OPTIONS_FCST=--output_chunks lead_time=50 --lead_dim_max_size 9
REFERENCE_TIME_PERIOD=1961-01-01 2018-12-30

## Labels
METRIC=rx1day
REGION=gn
TIMESCALE=annual-jan-to-dec
SIMILARITY_TEST=ks

# Shapefile for GEV parameters (mask invalid ocean points to avoid errors)
# SHAPE_OVERLAP=0.1
# SHAPEFILE=${PROJECT_DIR}/shapefiles/ne_110m_land_dissolved.shp

## Independence test options
INDEPENDENCE_OPTIONS=--confidence_interval 0.99 --n_resamples 1000
## Minimum lead time file options (independence file kwargs e.g., median min_lead over shapefile)
USE_MIN_INDEPENDENT_LEAD_FILE=0
MIN_IND_LEAD_OPTIONS=--min_lead 0


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
COVARIATE_BASE=2025
# Period for trend calculation (string converted to python)
GEV_TREND_PERIOD='[1961, 2026]'
# Dictionary of plot options for spatial analysis notebook (string converted to python)
PLOT_DICT='dict(metric="Rx1day", var="pr", var_name="Precipitation", units="mm / day", units_label="Precipitation [mm / day]", freq="${TIME_FREQ}", cmap=cmap_dict["pr"], cmap_anom=cmap_dict["pr_anom"], ticks=np.arange(0, 280, 25), ticks_anom=np.arange(-130, 130 + 20, 20), ticks_anom_std=np.arange(-22.5, 22.5 + 5, 5), ticks_anom_pct=np.arange(-110, 110 + 20, 20), ticks_anom_ratio=np.arange(-0.15, 5 + 0.5, 0.5), ticks_trend=np.arange(-22.5, 22.5 + 5, 5), ticks_param_trend={"location": np.arange(-2, 2.5, 0.5), "scale": np.arange(-0.5, 0.51, 0.1)}, ticks_soft_record=np.arange(0, 50 + 10, 10), cbar_extend="max", acs_map_plot_kwargs=dict(name="ncra_regions", mask_not_australia=False, agcd_mask=False, figsize=[6.2, 4.6], xlim=(113, 153), ylim=(-43, -8.5)), map_plot_kwargs=dict(mask_not_australia=False, xlim=(112.5, 154.3), ylim=(-44.5, -9.6), xticks=np.arange(120, 155, 10), yticks=np.arange(-40, -0, 10)))'

#  Plot additive/multiplicative bias corrected metric in spatial analysis notebook (True/False)#
PLOT_ADDITIVE_BC=1
PLOT_MULTIPLICATIVE_BC=1
