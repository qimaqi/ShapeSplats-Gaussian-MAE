#!/bin/bash
#SBATCH --job-name=mae_seg_sh
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=./joblogs/mae_seg_full_e250_1k_%j.log      # Redirect stdout to a log file
#SBATCH --error=./joblogs/mae_seg_full_e250_1k_%j.error     # Redirect stderr to a separate error log file

# cuda


source /scratch_net/schusch/qimaqi/miniconda3/etc/profile.d/conda.sh
conda activate gaussian_mae
# gaussian_mae_test 
# export CUDA_HOME=$CONDA_PREFIX
# export CC=/scratch_net/schusch/qimaqi/install_gcc/bin/gcc-11.3.0
# export CXX=/scratch_net/schusch/qimaqi/install_gcc/bin/g++-11.3.0
export CC=/scratch_net/schusch/qimaqi/install_gcc_8_5/bin/gcc-8.5.0
export CXX=/scratch_net/schusch/qimaqi/install_gcc_8_5/bin/g++-8.5.0



export PARTANNO_ROOT="/usr/bmicnas01/data-biwi-01/qimaqi_data_bmicscratch/data/ws_dataset_bak/shapenet/shapenet_part/shapenetcore_partanno_segmentation_benchmark_v0_normal/"
export GS_ROOT="/usr/bmicnas01/data-biwi-01/qimaqi_data_bmicscratch/data/ws_dataset_bak/shapenet/shapenet_ply"
export PC_TO_GS_MAP="/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/3dv_2025/ShapeSplat-Gaussian_MAE/segmentation_gs/split_to_org_gs_map.json"



PARTANNO_ROOT=${PARTANNO_ROOT}
GS_ROOT=${GS_ROOT}
PRETRAIN_CKPT="The path to the pretrain checkpoint"
PC_TO_GS_MAP=${PC_TO_GS_MAP}
cd ..

python main.py \
    --partanno_root ${PARTANNO_ROOT}  \
    --gs_root ${GS_ROOT} \
    --pc_to_gs_map ${PC_TO_GS_MAP} \
    --log_dir ./enc_full_group_xyz_pre_1k \
    --attribute '["xyz","opacity","scale","rotation","sh"]' \
    --ckpts ${PRETRAIN_CKPT} \
    --npoint  1024 \
    --num_group 64 \
    # --soft_knn
    # --use_wandb
    # --norm_attribute
    # --group_attribute
    

