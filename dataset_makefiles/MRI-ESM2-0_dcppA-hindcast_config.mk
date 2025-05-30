# Configuration files for Makefile workflows

MODEL=MRI-ESM2-0
EXPERIMENT=dcppA-hindcast
BASE_PERIOD=1970-01-01 2019-12-31
BASE_PERIOD_TEXT=1970-2019
TIME_PERIOD_TEXT=196011-201911
STABILITY_START_YEARS=1960 1970 1980 1990 2000 2010
MODEL_IO_OPTIONS=--n_ensemble_files 10 --metadata_file /g/data/xv83/unseen-projects/code/dataset_config/dataset_dcpp.yml
MODEL_NINO_OPTIONS=--n_ensemble_files 10 --lon_bnds 190 240 --lat_dim latitude --lon_dim longitude --agg_y_dim y --agg_x_dim x --anomaly ${BASE_PERIOD} --anomaly_freq month
