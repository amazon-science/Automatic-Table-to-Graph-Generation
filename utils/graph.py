"""
    Relevant utilities for graph-preprocessing/post-processing
"""
import os
import numpy as np
from utils.data.func import convert_tensor_to_list_arrays
import pandas as pd
import json
import torch

def convert_to_float64(df):
    """
    Convert all columns in a pandas DataFrame to float64.

    Args:
        df: The pandas DataFrame.

    Returns:
        The pandas DataFrame with all columns converted to float64.
    """
    for column in df.columns:
        if df[column].dtype == np.float64:
            continue
        if df[column].dtype == np.float32 or df[column].dtype == np.int64 or df[column].dtype == np.int32:
            df[column] = df[column].astype(np.float64)
        else:
            df[column] = df[column].astype('category')
            df[column] = df[column].cat.codes
            df[column] = df[column].astype(np.float64)
    return df

def create_graph_edges(df, column1, column2, skip_col2_na = True):
    """
    Creates graph edges from two columns in a pandas DataFrame.

    Args:
        df: The pandas DataFrame.
        column1: The name of the first column.
        column2: The name of the second column.

    Returns:
        A DataFrame containing edges as tuples (source_index, target_index).
    """
    # Create a mapping from value to index for the unique values in each column
    value_to_index_1 = {value: index for index, value in enumerate(df[column1].unique())}
    if skip_col2_na:
        value_to_index_2 = {value: index for index, value in enumerate(df[column2][df[column2].notnull()].unique())}
    else:
        value_to_index_2 = {value: index for index, value in enumerate(df[column2].unique())}
    
    # Create a new column in the dataframe to store the index from the mapping for each column
    # Apply the mapping to the original columns to get the corresponding indices
    if skip_col2_na:
        index1 = df[column1][df[column2].notnull()].apply(lambda x: value_to_index_1[x])
        index2 = df[column2][df[column2].notnull()].apply(lambda x: value_to_index_2[x])
    else:
        index1 = df[column1].apply(lambda x: value_to_index_1[x])
        index2 = df[column2].apply(lambda x: value_to_index_2[x])
    
    return torch.from_numpy(index1.values), torch.from_numpy(index2.values)


def graphstorm_graph_construction(node_type, meta_data, cache_path = './tmp'):
    """
        Meta data should provide
        1. for each node_type, the number of unique values `num_nodes`
        2. for each node_type, which columns should be used as features
            this should be a key-val mapping (name_of_the_feat, corresponding column)
        3. edge_type: which should be like {(src_node_type, edge_type, dst_node_type): (src_ids, dst_ids)}
    """
    # general format
    data_json = {}
    data_json['version'] = 'gconstruct-v0.1'
    data_json['nodes'] = []
    data_json['edges'] = []

    node_list = []
    edge_list = []

    node_prefix_dict = {}
    for ntype in node_type:
        node_prefix_dict[ntype] = ntype[0]
    
    for ntype in node_type:
        node_dict = {}
        ## number of nodes
        num_nodes = meta_data[ntype]['num_nodes']
        node_id = torch.arange(num_nodes)
        str_node_ids = np.array([f'{node_prefix_dict[ntype]}{i}' for i in node_id.numpy()])
        node_dict['node_id'] = str_node_ids
        if meta_data[ntype].get('features', None):
            ## this should be a key-val pair
            for feat_name, col in meta_data[ntype]['features'].items():
                node_dict[feat_name] = convert_tensor_to_list_arrays(col)
        
        node_df = pd.DataFrame(node_dict)
        print(f'{ntype} nodes have: {node_df.columns} columns ......')
        node_list.append((ntype, node_df))
    
    for (src_ntype, etype, dst_ntype), (src_ids, dst_ids) in meta_data['edges'].items():
        edge_dict = {}
        str_src_ids = np.array([f'{node_prefix_dict[src_ntype]}{i}' for i in src_ids.numpy()])
        str_dst_ids = np.array([f'{node_prefix_dict[dst_ntype]}{i}' for i in dst_ids.numpy()])
        edge_dict['source_id'] = str_src_ids
        edge_dict['dest_id'] = str_dst_ids
    
        if meta_data[etype].get('features', None):
            for feat_name, col in meta_data[etype]['features'].items():
                edge_dict[feat_name] = convert_tensor_to_list_arrays(col)
        
        edge_df = pd.DataFrame(edge_dict)
        edge_list.append(((src_ntype, etype, dst_ntype), edge_df))
    
    node_base_path = os.path.join(cache_path, 'nodes')
    if not os.path.exists(node_base_path):
        os.makedirs(node_base_path)
    # save node data files
    node_file_paths = {}
    for (ntype, node_df) in node_list:
        node_file_path = os.path.join(node_base_path, ntype + '.parquet')
        node_df.to_parquet(node_file_path)
        node_file_paths[ntype]= [node_file_path]
        print(f'Saved {ntype} node data to {node_file_path}.')
    
    edge_base_path = os.path.join(cache_path, 'edges')
    if not os.path.exists(edge_base_path):
        os.makedirs(edge_base_path)
    # save edge data files
    edge_file_paths = {}
    for (canonical_etype, edge_df) in edge_list:
        src_ntype, etype, dst_ntype = canonical_etype
        edge_file_name = src_ntype + '_' + etype + '_' + dst_ntype
        edge_file_path = os.path.join(edge_base_path, edge_file_name + '.parquet')
        edge_df.to_parquet(edge_file_path)
        edge_file_paths[canonical_etype] = [edge_file_path]
        print(f'Saved {canonical_etype} edge data to {edge_file_path}')
    
    # generate node json object
    node_jsons = []
    for (ntype, node_df) in node_list:
        node_dict = {}
        node_dict['node_type'] = ntype
        node_dict['format'] = {'name': 'parquet'}       # In this example, we just use parquet
        node_dict['files'] = node_file_paths[ntype]
        
        labels_list = []
        feats_list = []
        # check all dataframe columns
        for col in node_df.columns:
            label_dict = {}
            feat_dict = {}
            if col == 'node_id':
                node_dict['node_id_col'] = col
            elif col == 'label':
                label_dict['label_col'] = col
                label_dict['task_type'] = 'classification'
                label_dict['split_pct'] = [0.8, 0.1, 0.1]
                label_dict['label_stats_type'] = 'frequency_cnt'
                labels_list.append(label_dict)
            elif col == 'text':
                feat_dict['feature_col'] = col
                feat_dict['feature_name'] = col
                feat_dict['transform'] = {"name": "tokenize_hf",
                                          "bert_model": "bert-base-uncased",
                                          "max_seq_length": 16}
                feats_list.append(feat_dict)
            else:
                feat_dict['feature_col'] = col
                feat_dict['feature_name'] = col
                # for this example, we do not have transform for features
                feats_list.append(feat_dict)
        # set up the rest fileds of this node type
        if feats_list:
            node_dict['features'] = feats_list
        if labels_list:
            node_dict['labels'] = labels_list
        
        node_jsons.append(node_dict)

    # generate edge json object
    edge_jsons = []
    for (canonical_etype, edge_df) in edge_list:
        edge_dict = {}
        edge_dict['relation'] = canonical_etype
        edge_dict['format'] = {'name': 'parquet'}       # In this example, we just use parquet
        edge_dict['files'] = edge_file_paths[canonical_etype]

        labels_list = []
        feats_list = []
        src_ntype, etype, dst_ntype = canonical_etype
        # check all dataframe columns
        for col in edge_df.columns:
            label_dict = {}
            feat_dict = {}
            if col == 'source_id':
                edge_dict['source_id_col'] = col
            elif col == 'dest_id':
                edge_dict['dest_id_col'] = col
            elif col == 'label':
                label_dict['task_type'] = 'link_prediction'     # In ACM data, we do not have this
                                                                # edge task. Here is just for demo
                label_dict['split_pct'] = [0.8, 0.1, 0.1]       # Same as the label_split filed.
                                                                # The split pct values are just for
                                                                # demonstration purpose.
                labels_list.append(label_dict)
            else:
                feat_dict['feature_col'] = col
                feat_dict['feature_name'] = col
                # for this example, we do not have transform for features
                feats_list.append(feat_dict)
        # set up the rest fileds of this node type
        if feats_list:
            edge_dict['features'] = feats_list
        if labels_list:
            edge_dict['labels'] = labels_list
        
        edge_jsons.append(edge_dict)
    
    data_json['nodes'] = node_jsons
    data_json['edges'] = edge_jsons
    json_file_path = os.path.join(cache_path, 'config.json')
    with open(json_file_path, 'w', encoding='utf-8') as f:
        json.dump(data_json, f, indent=4)









        


    
    


def create_gs_graph_from_dgl(graph, text_feat = None, output_path = None):
    # generate node dataframe: we use the graph node ids and node name as node_type
    node_list = []

    # extract the first letter of each node type name as the prefix
    node_prefix_dict = {}
    for ntype in graph.ntypes:
        node_prefix_dict[ntype] = ntype[0]

    for ntype in graph.ntypes:
        node_dict = {}
        # generate the id column
        node_ids = graph.nodes(ntype)
        # pad a prefix before each node id
        str_node_ids = np.array([f'{node_prefix_dict[ntype]}{i}' for i in node_ids.numpy()])
        
        node_dict['node_id'] = str_node_ids

        # generate the feature columns and label column
        if graph.nodes[ntype].data:
            for feat_name, val in graph.nodes[ntype].data.items():
                # Here we just hard code the 'label' string
                if feat_name == 'label':
                   # convert tensor to list of arrays for saving in parquet format
                    node_dict[feat_name] = convert_tensor_to_list_arrays(val)
                    continue
                # Here we assume all others are node features
                # convert tensor to list of arrays for saving in parquet format
                node_dict[feat_name] = convert_tensor_to_list_arrays(val)

        # generate the raw text features column
        if text_feat is not None:
            node_dict['text'] = text_feat[ntype]

        # generate the pandas DataFrame that combine ids, and, if have, features and labels
        node_df = pd.DataFrame(node_dict)
        print(f'{ntype} nodes have: {node_df.columns} columns ......')
        # add node type name and node dataframe as a tuple
        node_list.append((ntype, node_df))

    # genreate edge dataframe
    edge_list = []
    
    for src_ntype, etype, dst_ntype in graph.canonical_etypes:
        edge_dict = {}
        # generate the ids columns for both source nodes and destination nodes
        src_ids, dst_ids = graph.edges(etype=(src_ntype, etype, dst_ntype))
        # pad a prefix before each node id
        str_src_ids = np.array([f'{node_prefix_dict[src_ntype]}{i}' for i in src_ids.numpy()])
        str_dst_ids = np.array([f'{node_prefix_dict[dst_ntype]}{i}' for i in dst_ids.numpy()])
        edge_dict['source_id'] = str_src_ids
        edge_dict['dest_id'] = str_dst_ids
        
        # generate feature columns and label col
        if graph.edges[(src_ntype, etype, dst_ntype)].data:
            for feat_name, val in graph.edges[(src_ntype, etype, dst_ntype)].data.items():
                if feat_name == 'label':
                    # Here we just hard code the 'label' string
                    # convert tensor to list of arrays for saving in parquet format
                    edge_dict['label'] = convert_tensor_to_list_arrays(val)
                    continue
                # Here we assume all others are edge features
                # convert tensor to list of arrays for saving in parquet format
                edge_dict[feat_name] = convert_tensor_to_list_arrays(val)
            
        # generate the pandas DataFrame that combine ids, and, if have, features and labels
        edge_df = pd.DataFrame(edge_dict)
        # add canonical edge type name and edge dataframe as a tuple
        edge_list.append(((src_ntype, etype, dst_ntype), edge_df))
    
    # output raw data files
    node_base_path = os.path.join(output_path, 'nodes')
    if not os.path.exists(node_base_path):
        os.makedirs(node_base_path)
    # save node data files
    node_file_paths = {}
    for (ntype, node_df) in node_list:
        node_file_path = os.path.join(node_base_path, ntype + '.parquet')
        node_df.to_parquet(node_file_path)
        node_file_paths[ntype]= [node_file_path]
        print(f'Saved {ntype} node data to {node_file_path}.')

    edge_base_path = os.path.join(output_path, 'edges')
    if not os.path.exists(edge_base_path):
        os.makedirs(edge_base_path)
    # save edge data files
    edge_file_paths = {}
    for (canonical_etype, edge_df) in edge_list:
        src_ntype, etype, dst_ntype = canonical_etype
        edge_file_name = src_ntype + '_' + etype + '_' + dst_ntype
        edge_file_path = os.path.join(edge_base_path, edge_file_name + '.parquet')
        edge_df.to_parquet(edge_file_path)
        edge_file_paths[canonical_etype] = [edge_file_path]
        print(f'Saved {canonical_etype} edge data to {edge_file_path}')

    # generate node json object
    node_jsons = []
    for (ntype, node_df) in node_list:
        node_dict = {}
        node_dict['node_type'] = ntype
        node_dict['format'] = {'name': 'parquet'}       # In this example, we just use parquet
        node_dict['files'] = node_file_paths[ntype]

        labels_list = []
        feats_list = []
        # check all dataframe columns
        for col in node_df.columns:
            label_dict = {}
            feat_dict = {}
            if col == 'node_id':
                node_dict['node_id_col'] = col
            elif col == 'label':
                label_dict['label_col'] = col
                label_dict['task_type'] = 'classification'
                label_dict['split_pct'] = [0.8, 0.1, 0.1]
                label_dict['label_stats_type'] = 'frequency_cnt'
                labels_list.append(label_dict)
            elif col == 'text':
                feat_dict['feature_col'] = col
                feat_dict['feature_name'] = col
                feat_dict['transform'] = {"name": "tokenize_hf",
                                          "bert_model": "bert-base-uncased",
                                          "max_seq_length": 16}
                feats_list.append(feat_dict)
            else:
                feat_dict['feature_col'] = col
                feat_dict['feature_name'] = col
                # for this example, we do not have transform for features
                feats_list.append(feat_dict)
        # set up the rest fileds of this node type
        if feats_list:
            node_dict['features'] = feats_list
        if labels_list:
            node_dict['labels'] = labels_list
        
        node_jsons.append(node_dict)

    # generate edge json object
    edge_jsons = []
    for (canonical_etype, edge_df) in edge_list:
        edge_dict = {}
        edge_dict['relation'] = canonical_etype
        edge_dict['format'] = {'name': 'parquet'}       # In this example, we just use parquet
        edge_dict['files'] = edge_file_paths[canonical_etype]

        labels_list = []
        feats_list = []
        src_ntype, etype, dst_ntype = canonical_etype
        # check all dataframe columns
        for col in edge_df.columns:
            label_dict = {}
            feat_dict = {}
            if col == 'source_id':
                edge_dict['source_id_col'] = col
            elif col == 'dest_id':
                edge_dict['dest_id_col'] = col
            elif col == 'label':
                label_dict['task_type'] = 'link_prediction'     # In ACM data, we do not have this
                                                                # edge task. Here is just for demo
                label_dict['split_pct'] = [0.8, 0.1, 0.1]       # Same as the label_split filed.
                                                                # The split pct values are just for
                                                                # demonstration purpose.
                labels_list.append(label_dict)
            else:
                feat_dict['feature_col'] = col
                feat_dict['feature_name'] = col
                # for this example, we do not have transform for features
                feats_list.append(feat_dict)
        # set up the rest fileds of this node type
        if feats_list:
            edge_dict['features'] = feats_list
        if labels_list:
            edge_dict['labels'] = labels_list
        
        edge_jsons.append(edge_dict)
        
    # generate the configuration JSON file
    data_json = {}
    data_json['version'] = 'gconstruct-v0.1'
    data_json['nodes'] = node_jsons
    data_json['edges'] = edge_jsons
        
    # output configration JSON
    json_file_path = os.path.join(output_path, 'config.json')
        
    with open(json_file_path, 'w', encoding='utf-8') as f:
        json.dump(data_json, f, indent=4)