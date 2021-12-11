#!/bin/bash
#SBATCH --job-name=medico_pointrend            # create a short name for your job
#SBATCH --output=/home/nero/MediaEval2021/Medico/Medico/slurm2.out      # create a output file
#SBATCH --error=/home/nero/MediaEval2021/Medico/Medico/slurm2.err       # create a error file
#SBATCH --partition=batch          # choose partition
#SBATCH --gres=gpu:2              # gpu count
#SBATCH --ntasks=1                 # total number of tasks across all nodes
#SBATCH --nodes=1                  # node count
#SBATCH --cpus-per-task=8          # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --time=3-0:00:00

echo   Date              = $(date)
echo   Hostname          = $(hostname -s)
echo   Working Directory = $(pwd)
echo   Number of Nodes Allocated      = $SLURM_JOB_NUM_NODES
echo   Number of Tasks Allocated      = $SLURM_NTASKS
echo   Number of Cores/Task Allocated = $SLURM_CPUS_PER_TASK

# Reconfigure HOME_PATH
# pwd

cd ~/MediaEval2021/Medico/
source env_medico/bin/activate

cd ~/MediaEval2021/Medico/Medico

# wandb login ae25f573bd73fc9acbde38b14a97f302dc84288e

# config="configs/image_based.yaml"
# OMP_NUM_THREADS=4 python tools/train.py --config ${config}
# OMP_NUM_THREADS=4 python tools/train.py 
OMP_NUM_THREADS=4 python -m torch.distributed.launch --master_port 9842 --nproc_per_node=1 tools/train.py --config configs/resunet101_cbam_pointrend.yml
# python sleep.py