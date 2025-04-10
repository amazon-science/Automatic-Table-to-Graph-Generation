import sentence_transformers
import sys
import os
import nltk
import numpy as np
from utils.data.rdb import load_dbb_dataset_from_cfg_path_no_name
from dbinfer import DBBRDBDataset
import torch
import pandas as pd
from tqdm import tqdm

def npz_to_dataframe(npz_file_path):
    """
    Converts a .npz file to a pandas DataFrame.

    Args:
        npz_file_path (str): Path to the .npz file.

    Returns:
        pd.DataFrame: A DataFrame constructed from the .npz file.
    """
    # Load the .npz file
    npz_data = np.load(npz_file_path)

    # Check if the .npz file has named arrays
    if isinstance(npz_data, np.lib.npyio.NpzFile):
        # Create a dictionary with array names as keys and arrays as values
        data_dict = {key: npz_data[key] for key in npz_data.files}
        # Convert the dictionary to a pandas DataFrame
        df = pd.DataFrame(data_dict)
    else:
        raise ValueError("The provided file does not contain named arrays.")

    # Close the .npz file
    npz_data.close()

    return df

def load_pretrain_jtd_lm(lm_path):
    """
        Load a pre-trained deep join language model
        Args:
            lm_path: the path to the pre-trained model
        Returns:
            model: the loaded model
    """

    if not os.path.exists(lm_path):
        raise FileNotFoundError("The model path does not exist")
    model = sentence_transformers.SentenceTransformer(lm_path)
    return model

def analyze_column_values(df, column_name, context_length=512, df_name = ""):
    """
        Analyze the column values in the dataset
        Args:
            df: the dataset
            column_name: the column name to analyze
            context_length: the maximum length of the context
        Returns:
            truncated_sentence: the truncated sentence
    """
    # if exist, return the cache 
    ## embedding tupe 
    if len(df[column_name].shape) > 1:
        return ""
    if df[column_name].dtype == 'object':
        return ""
    if pd.core.dtypes.common.is_datetime64_any_dtype(df[column_name]):
        return ""
    value_counts = pd.Series(df[column_name].astype(str)).value_counts()
    sorted_values = value_counts.index.tolist()
    n = len(sorted_values)
    col = ', '.join(sorted_values)
    lengths = [len(str(value)) for value in sorted_values]
    max_len = max(lengths)
    min_len = min(lengths)
    avg_len = sum(lengths) / len(lengths)
    tokens = f"{column_name} contains {str(n)} values ({str(max_len)}, {str(min_len)}, {str(avg_len)}): {col}"
    tokens = nltk.word_tokenize(tokens)
    truncated_tokens = tokens[:context_length]
    truncated_sentence = ' '.join(truncated_tokens)
    return truncated_sentence


def join_discovery(dataset: DBBRDBDataset, model: sentence_transformers.SentenceTransformer):
    """
        Discover joins between tables
        Args:
            df_list: RDB dataset
        Returns:
            joins: a numpy array indicating the join availability between tables
    """
    meta_info = {}
    for table_name in dataset.tables.keys():
        df = dataset.tables[table_name]
        for col_name in df.keys():
            col_value = df[col_name]
            column_values = analyze_column_values(df, col_name)
            meta_info[column_values] = (table_name, col_name)
    
    similarity = {}
    total_iterations = len(meta_info) * (len(meta_info) - 1) // 2
    with tqdm(total=total_iterations, desc="Processing", unit="iteration") as pbar:
        for i, key in enumerate(meta_info.keys()):
            for j, key2 in enumerate(meta_info.keys()):
                if key == "" or key2 == "":
                    pbar.update(1)
                    continue
                if key != key2 and i < j:
                    table_name1, col_name1 = meta_info[key]
                    table_name2, col_name2 = meta_info[key2]
                    sentence1 = key
                    sentence2 = key2
                    embeddings1 = model.encode(sentence1, convert_to_tensor=True)
                    embeddings2 = model.encode(sentence2, convert_to_tensor=True)
                    cosine_score = torch.cosine_similarity(embeddings1, embeddings2, dim=0)
                    similarity[(table_name1, col_name1, table_name2, col_name2)] = cosine_score.item()
                    pbar.update(1)
    return similarity




    
    
