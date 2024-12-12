#!/bin/bash
#SBATCH --job-name=gaussian_mae_cls
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --output=./joblogs/gs_cls_enc_full_group_xyz_1k_nopre_%j.log
#SBATCH --error=./joblogs/gs_cls_enc_full_group_xyz_1k_nopre_%j.error
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem 128GB

source /scratch_net/schusch/qimaqi/miniconda3/etc/profile.d/conda.sh
conda activate gaussian_mae

cd ..
cd ..

python main.py \
    --config cfgs/fintune/finetune_modelnet10_enc_full_group_xyz_1k.yaml \
    --finetune_model \
    --exp_name modelnet_cls_enc_full_group_xyz_1k_nopre \
    --seed 0

