#!/bin/bash
#PBS -P wp00
#PBS -q normal
#PBS -l walltime=20:00:00
#PBS -l mem=20GB
#PBS -l storage=gdata/xv83+gdata/oi10+gdata/wp00+gdata/ia39
#PBS -l wd
#PBS -v metric

# Example: qsub -v metric=txx spatial_job.sh

/g/data/xv83/dbi599/miniconda3/envs/unseen/bin/python spatial.py ${metric} ${metric}_spatial.nc
