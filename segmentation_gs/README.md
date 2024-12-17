# ShapeSplat Segmentation Fintune

This section provides detailed instructions on how to conduct part segmentation experiments using ShapeSplat Gaussians. The process is divided into three main parts:
- **Dataset Processing**  
- **Experiments**  
- **Results**  




## Dataset Processing
Following the instructions in [DATA.md](../DATA.md), the **shapenetcore_partanno_segmentation_benchmark_v0_normal** dataset should already be downloaded and unzipped.  

To ensure a fair comparison with point cloud-based methods, we use the same labels and splits provided in the **shapenetcore_partanno_segmentation_benchmark_v0_normal** dataset.  For each Gaussian input, the features are mapped to the corresponding point label positions. The segmentation labels are then predicted at these **point cloud positions**.

However, due to some misnaming issues, it is necessary to create a mapping between the point clouds and the ShapeSplats representations. This can be done by running the `prepare_seg_gs.py` script:  

```python
splits_file_root_path = 'Specify the path to shapenetcore_partanno_segmentation_benchmark_v0_normal/train_test_split/'
org_gs_save_path = 'Specify the path to the ShapeSplats directory'
...
```


## Experiments

Similar to ModelNet finetuning, after pretraining, you can parse the checkpoints path in `partseg_job_enc_full_group_xyz_1k.sh` in  `sh_jobs/partseg_job_enc_full_group_xyz_1k.sh`. Before run the segmentation, you need to specify following parameters:
- **`PARTANNO_ROOT`** 

The path where you unzip the shapenetcore_partanno_segmentation_benchmark_v0_normal

- **`GS_ROOT`** 
The path where the gaussian splats is saved

- **`PRETRAIN_CKPT`** 
Specifies the path of the '.pth' file after pretraining


```bash
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

```
You can specify other parameters like `soft_knn`, `norm_attribute` and `group_attribute` like modelnet finetuning. Note that here instead of parsing config file for each experiments, the parameters for experiments is decided by different input arguments. The above example demonstrates using **all attributes** for encoding and **only xyz** for grouping. To experiment with other configurations, create your own bash file with the desired parameter settings.



### Additional Parameters for Part Segmentation

You can specify additional parameters, such as:   `soft_knn`, `norm_attribute` and `group_attribute`. These are similar to the parameters used in **ModelNet finetuning**.  

**Note:** Unlike ModelNet finetuning, where configuration files are parsed for each experiment, the parameters for part segmentation experiments are directly controlled via input arguments. Ensure the desired parameters are set when running the script.


## Results:
The experiments results are saved in 'experiments/part_seg/log_dir', it include the checkpoints and the logs, you can check the pt.txt in logs and search for keyword best, then you can find the reported `Best accuracy`, `Best class avg mIOU` and `Best inctance avg mIOU` which reported in the paper.

### Viewing Experiment Results

The experiment results are saved in the directory `experiments/part_seg/log_dir`. This directory includes both the checkpoints and log files.  

To find the reported metrics, open the `pt.txt` file located in the logs directory and search for the keyword **"best"**. You will find the following metrics as reported in the paper:  
- **Best accuracy**  
- **Best class average mIOU**  
- **Best instance average mIOU**  

If you want to save results for visualization, you can store the `seg_pred` and `gs_data` array by modifying line 258 in the main function.