## Pasha Bulker analysis

rx2day (whole region)
```
/g/data/xv83/dbi599/miniconda3/envs/unseen/bin/fileio /g/data/zv2/agcd/v1-0-3/precip/total/r005/01day/agcd_v1_precip_total_r005_daily_*.nc /g/data/xv83/unseen-projects/outputs/pasha/data/rx2day_AGCD-CSIRO_1901-2024_A-AUG_pasha.nc --lat_bnds -34 -32 --lon_bnds 150 153 --variables pr --spatial_agg weighted_mean --rolling_sum_window 2 --shapefile /g/data/xv83/unseen-projects/outputs/pasha/shapefiles/pasha_area.shp --time_freq A-AUG --time_agg max --input_freq D --units_timing middle --reset_times --time_agg_min_tsteps 360 --verbose --time_agg_dates --units pr='mm day-1' --metadata_file /home/599/dbi599/unseen-projects/dataset_config/dataset_agcd_daily.yml
```

rx1day (LGAs)
```
/g/data/xv83/dbi599/miniconda3/envs/unseen/bin/fileio /g/data/zv2/agcd/v1-0-3/precip/total/r005/01day/agcd_v1_precip_total_r005_daily_*.nc /g/data/xv83/unseen-projects/outputs/pasha/data/rx1day_AGCD-CSIRO_1901-2024_YE-AUG_pasha-lgas.nc --lat_bnds -34 -32 --lon_bnds 150 153 --variables pr --spatial_agg weighted_mean --shapefile /g/data/xv83/unseen-projects/outputs/pasha/shapefiles/pasha_lgas.shp --shp_header LGA_NAME22 --time_freq YE-AUG --time_agg max --input_freq D --units_timing middle --reset_times --time_agg_min_tsteps 360 --verbose --time_agg_dates --units pr='mm day-1' --metadata_file /home/599/dbi599/unseen-projects/dataset_config/dataset_agcd_daily.yml
```

rx1day (Croudace Bay)
```
/g/data/xv83/dbi599/miniconda3/envs/unseen/bin/fileio /g/data/zv2/agcd/v1-0-3/precip/total/r005/01day/agcd_v1_precip_total_r005_daily_*.nc /g/data/xv83/unseen-projects/outputs/pasha/data/rx1day_AGCD-CSIRO_1901-2024_YE-AUG_croudace-bay.nc --point_selection -33 151.65 --variables pr --time_freq YE-AUG --time_agg max --input_freq D --units_timing middle --reset_times --time_agg_min_tsteps 360 --verbose --time_agg_dates --units pr='mm day-1' --metadata_file /home/599/dbi599/unseen-projects/dataset_config/dataset_agcd_daily.yml
```
