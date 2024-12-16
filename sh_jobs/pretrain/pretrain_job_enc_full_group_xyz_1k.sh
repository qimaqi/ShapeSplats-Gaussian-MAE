#!/bin/bash
#SBATCH --job-name=gaussian_mae_pretrain
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --output=./joblogs/gs_mae_enc_full_group_xyz_1k_%j.log
#SBATCH --error=./joblogs/gs_mae_enc_full_group_xyz_1k_%j.error
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem 128GB

source /scratch_net/schusch/qimaqi/miniconda3/etc/profile.d/conda.sh
conda activate gaussian_mae

cd ..
cd ..

echo "Job ID: $SLURM_JOBID"
echo "Time: $(date)"

python main.py \
    --config cfgs/pretrain/pretrain_enc_full_group_xyz_1k.yaml \
    --exp_name gaussian_mae_enc_full_group_xyz_1k \
    --num_workers=8 \
    # --resume 
