#!/bin/bash
#PBS -P xv83
#PBS -q normal
#PBS -l walltime=15:00:00
#PBS -l mem=20GB
#PBS -l storage=gdata/xv83+gdata/ia39
#PBS -l wd

# Example: qsub independence_job.sh

__conda_setup="$('/g/data/xv83/dbi599/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/g/data/xv83/dbi599/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/g/data/xv83/dbi599/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/g/data/xv83/dbi599/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup

conda activate unseen


models=('BCC-CSM2-MR' 'CAFE' 'CMCC-CM2-SR5' 'CanESM5' 'EC-Earth3' 'IPSL-CM6A-LR' 'MIROC6' 'MPI-ESM1-2-HR' 'MRI-ESM2-0' 'NorCPM1')

for model in "${models[@]}"; do
    infile=`ls /g/data/xv83/unseen-projects/outputs/bias/data/txx_${model}-*AUS300i.nc`
    outfile=`echo ${infile} | sed s:txx:independence-txx:`
    command="independence ${infile} tasmax ${outfile} --confidence_interval 0.99 --n_resamples 1000"
    echo ${command}
    ${command}
done

for model in "${models[@]}"; do
    infile=`ls /g/data/xv83/unseen-projects/outputs/bias/data/rx1day_${model}-*AUS300i.nc`
    outfile=`echo ${infile} | sed s:rx1day:independence-rx1day:`
    command="independence ${infile} pr ${outfile} --confidence_interval 0.99 --n_resamples 1000"
    echo ${command}
    ${command}
done


