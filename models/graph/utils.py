"""
    Exploration in terms of linear gnn-based oracle
    not used in this paper
"""
import numpy as np
import pandas as pd
from dbinfer.cli import construct_graph, preprocess
import os

def load_pqt_or_npz(filename):
    if filename.endswith('.pqt'):
        return pd.read_parquet(filename)
    elif filename.endswith('.npz'):
        return np.load(filename)
    elif filename.endswith('.npy'):
        return np.load(filename)
    else:
        raise ValueError(f"Unsupported file format: {filename}")
    
    

def construct_graph_for_oracle(dataset_id, schema_path, method):
    """
        Construct the graph for the oracle model
        Parameters:
            dataset_id: str
                The dataset id, for example, "MAG"
            schema_path: str
                The path to the schema file
    """
    current_oracle_backup_path = os.path.dirname(schema_path)
    preprocessed_file_paths = preprocess(dataset_id, schema_path, "transform", os.path.join(current_oracle_backup_path, 'preprocessed'), None)
    output_path = construct_graph(preprocessed_file_paths, method, os.path.join(current_oracle_backup_path, 'graph'), None) 
    return output_path


def construct_dfs_for_oracle(dataset_id, schema_path, dfs_layer = 2):
    current_oracle_backup_path = os.path.dirname(schema_path)
    pre_dfs_config = f'configs/transform/pre-dfs.yaml'
    preprocessed_file_paths = preprocess(dataset_id, schema_path, "transform", os.path.join(current_oracle_backup_path, 'preprocessed'), pre_dfs_config)
    dfs_config = f'configs/dfs/dfs-{dfs_layer}.yaml'
    pre_dfs_file_paths = preprocess(dataset_id, preprocessed_file_paths, "dfs", os.path.join(current_oracle_backup_path, 'post_dfs'), dfs_config)
    post_dfs_config = f'configs/transform/post-dfs.yaml'
    post_dfs_file_paths = preprocess(dataset_id, pre_dfs_file_paths, "transform", os.path.join(current_oracle_backup_path, f'{dataset_id}-dfs-{dfs_layer}'), post_dfs_config)
    return post_dfs_file_paths
    
