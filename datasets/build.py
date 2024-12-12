from utils import registry


DATASETS = registry.Registry('dataset')


def build_dataset_from_cfg(cfg, default_args = None):
    """
    Build a dataset, defined by `dataset_name`.
    Args:
        cfg (eDICT): 
    Returns:
        Dataset: a constructed dataset specified by dataset_name.
    """
    # print("cfg", cfg)
    # print("default_args", default_args)
    return DATASETS.build(cfg, default_args = default_args)


