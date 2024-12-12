import os
import yaml
WORKING_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


""" TYPE the PAHT you unzip the SHAPESPLATs and MODELSPLATS """
SHAPESPLAT_DIR = None 
MODELSPLAT_DIR = None


# start overwrite
assert SHAPESPLAT_DIR is not None, "Please set the SHAPESPLIT_DIR"
assert os.path.exists(SHAPESPLAT_DIR), "SHAPESPLIT_DIR does not exist"

assert MODELSPLAT_DIR is not None, "Please set the MODELSPLIT_DIR"
assert os.path.exists(MODELSPLAT_DIR), "MODELSPLIT_DIR does not exist"



SHAPENET_SPLIT_DIR = os.path.join(WORKING_DIR, 'datasets', 'shapenet_split')
MODELNET_SPLIT_DIR = os.path.join(WORKING_DIR, 'datasets', 'models_split')

# overwrite ShapeSplat
ShapeNet55GS_dir = os.path.join(WORKING_DIR, 'cfgs', 'dataset_configs','ShapeNet55GS.yaml')
with open(ShapeNet55GS_dir, 'r') as f:
    dataset_config = yaml.load(f, Loader=yaml.FullLoader)
dataset_config['DATA_PATH'] = SHAPENET_SPLIT_DIR
dataset_config['GS_PATH'] = SHAPESPLAT_DIR

with open(ShapeNet55GS_dir, 'w') as f:
    yaml.dump(dataset_config, f, default_flow_style=False)

# overwrite ModelSplat
ModelNet40_dir = os.path.join(WORKING_DIR, 'cfgs', 'dataset_configs','ModelNet40GS.yaml')
with open(ModelNet40_dir, 'r') as f:
    dataset_config = yaml.load(f, Loader=yaml.FullLoader)
dataset_config['DATA_PATH'] = MODELNET_SPLIT_DIR
dataset_config['GS_PATH'] = MODELSPLAT_DIR

with open(ModelNet40_dir, 'w') as f:
    yaml.dump(dataset_config, f, default_flow_style=False)

ModelNet10_dir = os.path.join(WORKING_DIR, 'cfgs', 'dataset_configs','ModelNet10GS.yaml')
with open(ModelNet10_dir, 'r') as f:
    dataset_config = yaml.load(f, Loader=yaml.FullLoader)

dataset_config['DATA_PATH'] = MODELNET_SPLIT_DIR
dataset_config['GS_PATH'] = MODELSPLAT_DIR