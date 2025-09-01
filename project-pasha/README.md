## Pasha Bulker analysis

```
/g/data/xv83/dbi599/miniconda3/envs/unseen/bin/fileio /g/data/xv83/agcd-csiro/precip/daily/precip-total_AGCD-CSIRO_r005_*_daily.nc /g/data/xv83/unseen-projects/outputs/pasha/data/rx2day_AGCD-CSIRO_1901-2024_A-AUG_pasha.nc --lat_bnds -34 -32 --lon_bnds 150 153 --variables pr --spatial_agg weighted_mean --rolling_sum_window 2 --shapefile /g/data/xv83/unseen-projects/outputs/pasha/shapefiles/pasha_area.shp --time_freq A-AUG --time_agg max --input_freq D --units_timing middle --reset_times --time_agg_min_tsteps 360 --verbose --time_agg_dates --units pr='mm day-1' --metadata_file /home/599/dbi599/unseen-projects/dataset_config/dataset_agcd_daily.yml

```

