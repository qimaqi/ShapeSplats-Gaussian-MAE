# DATASET Preparation

This section provides instructions on downloading and preprocessing the following datasets:  
- **ShapeSplat Dataset**  
- **ModelSplat Dataset**  
- **ShapeNet-Part Dataset**  

## ShapeSplat pretrain dataset
1. **Request Dataset Access**  
   Obtain access to the ShapeNet dataset from the [Hugging Face page](https://huggingface.co/datasets/ShapeNet/ShapeSplatsV1). Approval may take approximately **1–2 days**.  

2. **Download the Dataset**  
   Once access is granted, you can download the dataset using the following steps:  

   ```sh
   mkdir -p gs_data/shapesplat
   cd gs_data/shapesplat
   
   huggingface-cli download ShapeNet/ShapeSplatsV1 --token <Your Hugging Face Token> --repo-type dataset --local-dir .
   ```  

   Replace `<Your Hugging Face Token>` with your personal access token from Hugging Face. Also feel free to save the dataset to anywhere you want.

3. **Process the Dataset**
   After downloading the dataset, you will find multiple `.zip` files in the target folder. Use the provided script `./scripts/unzip_shapesplat.sh` to extract all the files like following structures.  The extracted files will follow the structure below, with all .ply files named in the format: `< {category_id}-{object_id}.ply>`.

```
shapesplat_ply
├── 04256520-81d3c178c611199831e05c4a367a9ebd.ply
├── 03001627-d388617a719f23a191d0dff7aea42471.ply
├── 03001627-2060f4d31feb73dd7762ba3a489df006.ply
├── 03991062-ad258c77538632867bc41009043d88b0.ply
├── 03691459-707f0e44e935dd55edfd593a4f114036.ply

``` 

   Update the script with the following values:  
   - **`<SOURCE_DIR>`**: Path to the folder where the downloaded `.zip` files are located.  
   - **`<DEST_DIR>`**: Path to the folder where you want the data to be extracted.  

   Run the script to complete the unzipping process.


```     

   # Directory containing the .zip files
   SOURCE_DIR=../gs_data/shapesplat/
   # Directory where all extracted files will be merged
   DEST_DIR=../gs_data/shapesplat/shapesplat_ply
   
``` 


3. **Prepare the Dataset config**
   The `ShapeNet55GS.yaml` file is located in the `cfgs/dataset_configs` directory. You can manually set the following paths:  

   - **`DATA_PATH`**: Set this to `datasets/shapenet_split`.  
   - **`GS_PATH`**: Set this to the directory where you extracted the `.ply` files.  

   Ensure both paths are correctly configured to avoid errors during dataset loading.  

   You can also use the code in `./scripts/overwrite_data_path.py` to do such overwrite:


## ModelSplat finetune dataset
1. **Download the Dataset**  
   Similariy, you can download the modelsplat in [ModelNetSplats Dataset Release](https://huggingface.co/datasets/ShapeSplats/ModelNet_Splats)

   ```sh
   mkdir -p gs_data/modelsplat
   cd gs_data/modelsplat
   
   huggingface-cli download ShapeSplats/ModelNet_Splats --token  <Your Hugging Face Token> --repo-type dataset --local-dir .
   ```
   Replace `<Your Hugging Face Token>` with your personal access token from Hugging Face. Also feel free to save the dataset to anywhere you want.

2. **Process the Dataset**
   After downloading there will be a lot of .zip file, we want to unzip to following data structure:

```
modelsplat
├── airplane
│   ├── train 
│   │   ├── airplane_0001
│   │   │   ├── point_cloud.ply
│   │   ├── airplane_0002
│   │   │   ├── point_cloud.ply
│   │   ├── .......
│   ├── test                 
│   │   ├── airplane_0627
│   │   │   ├── point_cloud.ply
│   │   ├── airplane_0002
│   │   │   ├── airplane_0628.ply
│   │   ├── .......
├── bathtub
├── .......

```

Using the provided script `./scripts/unzip_modelsplat.sh`, you only need to configure the following parameters to begin the unzipping process.  
   - **`<SOURCE_DIR>`**: Path to the folder where the downloaded `.zip` files are located.  
   - **`<DEST_DIR>`**: Path to the folder where you want the data to be extracted.  



3. **Prepare the Dataset config**
   The `ModelNet10GS.yaml` and `ModelNet40GS.yaml`  files are located in the `cfgs/dataset_configs` directory. You can manually set the following paths:  

   - **`DATA_PATH`**: Set this to `ShapeSplat-Gaussian_MAE/datasets/modelnet_split`.  
   - **`GS_PATH`**: Set this to the directory where you extracted the `.ply` files.  

   Ensure both paths are correctly configured to avoid errors during dataset loading.  

   You can also use the code in `./scripts/overwrite_data_path.py` to do such overwrite:



## ShaeSplat Part-Seg finetune dataset
1. **Download the Dataset** 
   
   You can officially download the Shape-Part annotations from the following [link](https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip).  

   If the official link is unavailable, you can also download the dataset from our [Hugging Face repository](https://huggingface.co/datasets/ShapeSplats/sharing/tree/main).  


   After unzipping shapenetcore_partanno_segmentation_benchmark_v0_normal.zip, you need to update the following environment variables:

   PARTANNO_ROOT
   **`GS_ROOT`** (Note: This should be the same as the GS_PATH used for ShapeSplat)
   **`PC_ROOT`** (Note: This is the path where to `shape_data`  the pointcloud annotation)
   This version is more concise and clearly lists the steps and notes required for setting up the environment.