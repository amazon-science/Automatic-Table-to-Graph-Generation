import os.path as osp
from dbinfer_bench import DBBRDBDataset

ALL_DATASETS = ["MAG", "IEEE", "AVS", "DIG", "OBS", "SE", "F1", "CLI", "MVLS", "HC", "RR", "ESCI", "AMAG", "STE"]
name_id_mapping = {
    "MAG": "mag",
    "AMAG": "magm",
    "IEEE": "ieee-cis", 
    "AVS": "avs",
    "DIG": "diginetica",
    "OBS": "outbrain-small",
    "F1": "f1",
    "CLI": "clinical",
    "MVLS": "movielens",
    "STE": "stackexchange",
    "RR": "retailrocket",
    "ESCI": "esci"
}


def load_dbb_dataset(dataset_name: str, cache_path: str):
    assert dataset_name in ALL_DATASETS
    data_id = name_id_mapping[dataset_name]
    full_path = osp.join(cache_path, data_id)
    return DBBRDBDataset(full_path), data_id

def load_dbb_dataset_from_cfg_path(dataset_name:str, cfg_path: str):
    assert dataset_name in ALL_DATASETS
    data_id = name_id_mapping[dataset_name]
    return DBBRDBDataset(cfg_path), data_id

def load_dbb_dataset_from_obj(obj, cfg_path: str):
    return DBBRDBDataset(path=cfg_path, metadata=obj)

def load_dbb_dataset_from_cfg_path_no_name(cfg_path: str):
    return DBBRDBDataset(cfg_path)

def get_data_id(dataset_name: str):
    assert dataset_name in ALL_DATASETS
    return name_id_mapping[dataset_name]


