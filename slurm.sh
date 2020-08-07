#!/bin/bash
#
#SBATCH  --mail-type=ALL                      # mail configuration: NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH  --output=/itet-stor/barthc/net_scratch/logs/log%j.log      # where to store the output ( %j is the JOBID )
#SBATCH  --cpus-per-task=1                    # Use 16 CPUS
#SBATCH  --gres=gpu:1                         # Use 1 GPUS
#SBATCH  --mem=32G                            # use 32GB
#SBATCH  --account=tik                        # we are TIK!

#eval "$(pyenv init -)"
#eval "$(pyenv virtualenv-init -)"

export TEMP="/itet-stor/barthc/net_scratch/tmp"
export TMPDIR="/itet-stor/barthc/net_scratch/tmp"

source conda activate permafrost
#
echo Running on host: `hostname`
echo In directory: `pwd`
echo Starting on: `date`
echo SLURM_JOB_ID: $SLURM_JOB_ID
#
# binary to execute
cd /home/barthc/MasterThesis/permafrostanalytics
#python -m pyinstrument -r text -o profile.o MAOnlineAlpineLaboratory/dataset_profiler.py -p /itet-stor/barthc/net_scratch/data --tmp_dir /itet-stor/barthc/net_scratch/user_dir/tmp --use_frozen
python MAOnlineAlpineLaboratory/classifier.py -p /itet-stor/barthc/net_scratch/data --tmp_dir /itet-stor/barthc/net_scratch/user_dir/tmp --use_frozen
#python MAOnlineAlpineLaboratory/classifier_dataset-freezer.py -p /itet-stor/barthc/net_scratch/data --tmp_dir /itet-stor/barthc/net_scratch/user_dir/tmp --use_frozen
echo finished at: `date`
exit 0;

