#!/bin/bash
#SBATCH --job-name=gaussian_mae_full_1k
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --output=./joblogs/gs_mae_full_1k_%j.log
#SBATCH --error=./joblogs/gs_mae_full_1k_%j.error
#SBATCH --time=48:00:00
#SBATCH --nodelist=bmicgpu07
#SBATCH --cpus-per-task=4
#SBATCH --mem 128GB

source /scratch_net/schusch/qimaqi/miniconda3/etc/profile.d/conda.sh
conda activate gaussian_mae
# gaussian_mae_test 
# export CUDA_HOME=$CONDA_PREFIX
# export CC=/scratch_net/schusch/qimaqi/install_gcc/bin/gcc-11.3.0
# export CXX=/scratch_net/schusch/qimaqi/install_gcc/bin/g++-11.3.0
export CC=/scratch_net/schusch/qimaqi/install_gcc_8_5/bin/gcc-8.5.0
export CXX=/scratch_net/schusch/qimaqi/install_gcc_8_5/bin/g++-8.5.0

# export PATH=$CUDA_HOME/bin:$PATH
# export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH

# gs_mae
# export CUDA_LAUNCH_BLOCKING=1
# cd /usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/3dv_2025/ShapeSplat-Gaussian_MAE
cd ..
cd ..

PRETRAIN_CKPT=/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/3dv_2025/ShapeSplat-Gaussian_MAE/experiments/pretrain_enc_full_group_full_1k/pretrain/gaussian_mae_enc_full_group_full_1k/ckpt-last.pth

# check if PRETRAIN_CKPT exists
if [ ! -f "$PRETRAIN_CKPT" ]; then
    echo "$PRETRAIN_CKPT does not exist."
    exit 1
fi


python main.py \
    --config cfgs/fintune/finetune_modelnet40_enc_full_group_xyz_1k.yaml \
    --finetune_model \
    --exp_name modelnet40_cls_enc_full_group_xyz_1k \
    --seed 0 \
    --ckpts ${PRETRAIN_CKPT}
    
    # replace with ckpts from pretrain

