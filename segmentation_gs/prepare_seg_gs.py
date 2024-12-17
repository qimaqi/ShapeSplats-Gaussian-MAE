import os 
import shutil
from tqdm import tqdm 
import numpy as np
import json 

# shuffled_train_file_list, shuffled_val_file_list.json, shuffled_test_file_list.json
splits_file_lists = ['shuffled_train_file_list.json', 'shuffled_val_file_list.json', 'shuffled_test_file_list.json']
splits_file_root_path = '/usr/bmicnas01/data-biwi-01/qimaqi_data_bmicscratch/data/ws_dataset_bak/shapenet/shapenet_part/shapenetcore_partanno_segmentation_benchmark_v0_normal/train_test_split/'
org_gs_save_path = '/usr/bmicnas01/data-biwi-01/qimaqi_data_bmicscratch/data/ws_dataset_bak/shapenet/shapenet_ply'

# create mapping:
split_to_org_gs_map = {}


for split_i in splits_file_lists:
    # first find missing data
    split_i_path = os.path.join(splits_file_root_path, split_i)

    with open(split_i_path) as f:
        original_data = json.load(f)

    """ then we copy the object to different category """
    for obj_i_path in tqdm(original_data):
        cat_id = obj_i_path.split('/')[-2]
        obj_id = obj_i_path.split('/')[-1]
        key = f'{cat_id}-{obj_id}'
        split_to_org_gs_map[key] = None
        # check if the file exists
        for exist_id in os.listdir(org_gs_save_path):
            exist_cat_id = exist_id.split('-')[0]
            exist_obj_id = exist_id.split('-')[1].split('.')[0]
            if exist_obj_id == obj_id:
                if cat_id == exist_cat_id:
                    split_to_org_gs_map[key] = f'{cat_id}-{exist_obj_id}.ply'
                else:
                    split_to_org_gs_map[key] = exist_id

        if split_to_org_gs_map[key] is None:
            print("missing", obj_i_path, "in gs folder") 
      

with open('./split_to_org_gs_map.json', 'w') as f:
    json.dump(split_to_org_gs_map, f)

