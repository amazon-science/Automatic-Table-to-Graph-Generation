import os
import os.path as osp
import string
import random
import datetime
import yaml
import numpy as np
from utils.training import extract_train_val_test_id_from_objects, train_val_test_split
from dbinfer.task_construct_utils import train_val_test_split_by_temporal
from utils.data.rdb import load_dbb_dataset_from_cfg_path_no_name
import pandas as pd
import yaml
import datetime
from models.llm.gconstruct import analyze_dataframes
import string
import random
import duckdb
import shutil
import typer
from ogb.nodeproppred import DglNodePropPredDataset

def generate_random_string(length):
    # Define the character set (letters and digits)
    characters = string.ascii_letters + string.digits
    
    # Generate the random string
    return ''.join(random.choice(characters) for _ in range(length))

def map_keys(original_dict, key_mapping):
    """Creates a new dictionary with keys remapped according to the provided mapping.

    Args:
        original_dict (dict): The dictionary whose keys need to be remapped.
        key_mapping (dict): A dictionary where keys are the original keys, 
                             and values are the new keys.

    Returns:
        dict: A new dictionary with the remapped keys.
    """
    new_dict = {}
    for key, value in original_dict.items():
        new_key = key_mapping.get(key, key)  # Use original key if no mapping exists
        new_dict[new_key] = value
    return new_dict


def main(dataset: str = typer.Argument("MAG", help="name of the dataset to be processed"), 
         dataset_path: str = typer.Argument("/home/ubuntu/data/relbench", help="path to the dataset")):
    # dataset_path = "datasets"
    # dataset_path = "newdatasets"
    if dataset == "MAG":
        print("Processing MAG dataset")
        
        original_mag = DglNodePropPredDataset(name="ogbn-mag", root=f"{dataset_path}/mag/raw")         
        ## generate the dataset schema
        year_tensor = original_mag[0][0].ndata["year"]['paper'].reshape(-1)
        full_path = f"{dataset_path}/mag/raw/ogbn_mag"
        original_data = osp.join(full_path, "data", "paper.npz")
        file = np.load(original_data)
        dict_file = dict(file)
        year_np = year_tensor.numpy()
        dict_file['year'] = year_np
        ## construct the expert schema
        expert_path = f"{dataset_path}/mag/expert"
        ## copy the data file from raw path to expert path
        os.makedirs(osp.join(expert_path, "data"), exist_ok=True)
        shutil.copytree(osp.join(full_path, "data"), osp.join(expert_path, "data"), dirs_exist_ok=True)
        shutil.copytree(osp.join(full_path, "cite"), osp.join(expert_path, "cite"), dirs_exist_ok=True)
        expert_yaml = """
dataset_name: mag
tables:
- columns:
  - dtype: float
    name: feat
  - dtype: category
    name: label
  - dtype: primary_key
    name: paperID
  - dtype: category
    name: year
  format: numpy
  name: Paper
  source: data/paper.npz
- columns:
  - dtype: foreign_key
    link_to: Paper.paperID
    name: paper_cite
  - dtype: foreign_key
    link_to: Paper.paperID
    name: paper_cited
  format: parquet
  name: Cites
  source: data/cites.pqt
- columns:
  - dtype: foreign_key
    link_to: FieldOfStudy.id
    name: field_of_study
  - dtype: foreign_key
    link_to: Paper.paperID
    name: paper
  format: parquet
  name: HasTopic
  source: data/has_topic.pqt
- columns:
  - dtype: foreign_key
    link_to: Author.id
    name: author
  - dtype: foreign_key
    link_to: Institution.id
    name: institution
  format: parquet
  name: AffiliatedWith
  source: data/affiliated_with.pqt
- columns:
  - dtype: foreign_key
    link_to: Author.id
    name: author
  - dtype: foreign_key
    link_to: Paper.paperID
    name: paper
  format: parquet
  name: Writes
  source: data/writes.pqt
tasks:
- columns:
  - dtype: float
    name: feat
  - dtype: category
    name: label
  - dtype: primary_key
    name: paperID
  - dtype: category
    name: year
  evaluation_metric: accuracy
  format: numpy
  name: venue
  source: venue/{split}.npz
  target_column: label
  target_table: Paper
  task_type: classification
- columns:
  - dtype: foreign_key
    link_to: Paper.paperID
    name: paper_cite
  - dtype: foreign_key
    link_to: Paper.paperID
    name: paper_cited
  evaluation_metric: mrr
  format: parquet
  name: cite
  source: cite/{split}.pqt
  target_column: paper_cited
  target_table: Cites
  task_type: retrieval
- columns:
  - dtype: float
    name: feat
  - dtype: category
    name: label
  - dtype: primary_key
    name: paperID
  - dtype: category
    name: year
  evaluation_metric: accuracy
  format: numpy
  name: year
  source: year/{split}.npz
  target_column: year
  target_table: Paper
  task_type: classification
"""
        expert_yaml = yaml.safe_load(expert_yaml)
        with open(f"{expert_path}/metadata.yaml", "w") as f:
            yaml.dump(expert_yaml, f)
        ## save this to both old and expert schema
        expert_path = f"{dataset_path}/mag/expert"
        np.savez(osp.join(expert_path, "data", "paper.npz"), **dict_file)
        ## 5 tables paper, affiliated_with, cites, has_topic, writes
        ## 3 tasks venue, year, paper
        original_table = dict_file
        extract_train_val_test_id_from_objects(f"{expert_path}/venue", "paperID", original_table)
        extract_train_val_test_id_from_objects(f"{expert_path}/year", "paperID", original_table, stratify=year_np)
        ## generate the original schema
        old_path = f"{dataset_path}/mag/old"
        np.savez(osp.join(old_path, "data", "paper.npz"), **dict_file)
        extract_train_val_test_id_from_objects(f"{old_path}/venue", "paperID", original_table)
        extract_train_val_test_id_from_objects(f"{old_path}/year", "paperID", original_table, stratify=year_np)
        shutil.copytree(osp.join(full_path, "cite"), osp.join(old_path, "cite"), dirs_exist_ok=True)
        old_yaml = """
dataset_name: mag
tables:
  - name: Paper
    source: data/paper.npz
    format: numpy
    columns:
      - name: feat
        dtype: float
      - name: label
        dtype: category
      - name: paperID
        dtype: primary_key
      - name: year
        dtype: category
  - name: Cites
    source: data/cites.pqt
    format: parquet
    columns:
      - name: paper_cite
        dtype: foreign_key
        link_to: Paper.paperID
      - name: paper_cited
        dtype: foreign_key
        link_to: Paper.paperID
  - name: HasTopic
    source: data/has_topic.pqt
    format: parquet
    columns:
      - name: field_of_study
        dtype: category
      - name: paper_name
        dtype: foreign_key
        link_to: Paper.paperID
  - name: AffiliatedWith
    source: data/affiliated_with.pqt
    format: parquet
    columns:
      - name: author
        dtype: category
      - name: institution
        dtype: category
  - name: Writes
    source: data/writes.pqt
    format: parquet
    columns:
      - name: paper_writer
        dtype: category
      - name: arxiv_id
        dtype: foreign_key
        link_to: Paper.paperID
tasks:
  - name: venue
    source: venue/{split}.npz
    format: numpy
    columns:
      - name: feat
        dtype: float
      - name: label
        dtype: category
      - name: paperID
        dtype: primary_key
      - name: year
        dtype: category
    evaluation_metric: accuracy
    target_column: label
    target_table: Paper
    task_type: classification
  - name: cite
    source: cite/{split}.pqt
    format: parquet
    columns:
      - name: paper_cite
        dtype: foreign_key
        link_to: Paper.paperID
      - name: paper_cited
        dtype: foreign_key
        link_to: Paper.paperID
    evaluation_metric: mrr
    target_column: paper_cited
    target_table: Cites
    task_type: retrieval
  - name: year 
    source: year/{split}.npz
    format: numpy
    columns:
      - name: feat
        dtype: float
      - name: label
        dtype: category
      - name: paperID
        dtype: primary_key
      - name: year
        dtype: category
    evaluation_metric: accuracy
    target_column: year
    target_table: Paper
    task_type: classification
"""
        with open(f"{old_path}/metadata.yaml", "w") as f:
            yaml.dump(yaml.safe_load(old_yaml), f)
        ## change the paper to paper_name in has_topic
        has_topic = pd.read_parquet(osp.join(expert_path, "data", "has_topic.pqt"))
        has_topic.rename(columns={"paper": "paper_name"}, inplace=True)
        has_topic.to_parquet(osp.join(old_path, "data", "has_topic.pqt"))
        ## change the paper_writer to author in writes
        writes = pd.read_parquet(osp.join(expert_path, "data", "writes.pqt"))
        writes.rename(columns={"author": "paper_writer"}, inplace=True)
        ## change the arxiv_id to paper in writes
        writes.rename(columns={"paper": "arxiv_id"}, inplace=True)
        writes.to_parquet(osp.join(old_path, "data", "writes.pqt"))
        ## affilited with and cites no change, directly copy to old schema
        affiliated_with = pd.read_parquet(osp.join(expert_path, "data", "affiliated_with.pqt"))
        cites = pd.read_parquet(osp.join(expert_path, "data", "cites.pqt"))
        affiliated_with.to_parquet(osp.join(old_path, "data", "affiliated_with.pqt"))
        cites.to_parquet(osp.join(old_path, "data", "cites.pqt"))
        information = analyze_dataframes({'Table paper': dict_file, 'Table Cites': cites, 'Table HasTopic': has_topic, 'Table AffiliatedWith': affiliated_with, 'Table Writes': writes}, k = 5)
        with open(f"{dataset_path}/mag/information.txt", "w") as f:
            f.write(information)
    elif dataset == "IEEE-CIS":
        print("Processing IEEE-CIS dataset")
        transaction_df =  pd.read_csv(f"{dataset_path}/ieeecis/raw/train_transaction.csv")
        identity_df = pd.read_csv(f"{dataset_path}/ieeecis/raw/train_identity.csv")
        meta_info = {}
        ## process original df data
        ## drop transactionDT
        transaction_df['index'] = transaction_df.index
        transaction_df.drop(columns=["TransactionDT"], inplace=True)
        transaction_df['TransactionAmt'] = np.log1p(transaction_df['TransactionAmt'])
        transaction_df.rename(columns={"ProductCD": "ProductCode"}, inplace=True) 
        card_rename = {f"card{i}": f"card_meta_info_{i}" for i in range(1, 7)}
        transaction_df.rename(columns=card_rename, inplace=True)
        transaction_df.rename(columns={"addr1": "purchaser billing region", "addr2": "purchaser billing country"}, inplace=True)
        # transaction_df.rename(columns={"dist1": "distance1"}, inplace=True)
        transaction_df.rename(columns={"P_emaildomain": "purchaser email domain", "R_emaildomain": "recipient email domain"}, inplace=True)
        counting_rename = {f"C{i}": f"payment_card_related_counting_{i}" for i in range(1, 15)}
        timedelta_rename = {f"D{i}": f"timedelta_{i}" for i in range(1, 16)}
        transaction_df.rename(columns=counting_rename, inplace=True)
        transaction_df.rename(columns=timedelta_rename, inplace=True)
        match_rename = {f"M{i}": f"match_{i}" for i in range(1, 10)}
        transaction_df.rename(columns=match_rename, inplace=True)
        transaction_categorical_columns = ["ProductCode", "card_meta_info_1", "card_meta_info_2", "card_meta_info_3", "card_meta_info_4", "card_meta_info_5", "card_meta_info_6", "purchaser billing region", "purchaser billing country", "purchaser email domain", "recipient email domain", "match_1", "match_2", "match_3", "match_4", "match_5", "match_6", "match_7", "match_8", "match_9"]
        id_rename_dict = {f"id_{i:02d}": f"identity_{i}_info" for i in range(1, 39)}
        identity_df.rename(columns=id_rename_dict, inplace=True)
        identity_categorical_columns = [f"identity_{i}_info" for i in range(12, 39)]
        identity_categorical_columns = identity_categorical_columns + ["DeviceType", "DeviceInfo"]
        meta_info['dataset_name'] = "ieeecis"
        meta_info['tables'] = []
        transaction_dict = {}
        transaction_value = {}
        identity_dict = {}
        identity_value = {}
        transaction_dict["name"] = "Transaction"
        transaction_dict["source"] = "data/transaction.npz"
        transaction_dict["format"] = "numpy"
        transaction_dict["columns"] = []
        for c in transaction_categorical_columns:
            col = {}
            col["name"] = c
            col["dtype"] = "category"
            transaction_dict["columns"].append(col)
            transaction_value[c] = transaction_df[c].values
        transaction_dict["columns"].append({"name": "TransactionID", "dtype": "primary_key"})
        transaction_value["TransactionID"] = transaction_df["TransactionID"].values
        transaction_dict["columns"].append({"name": "isFraud", "dtype": "category"})
        transaction_value["isFraud"] = transaction_df["isFraud"].values
        transaction_dict["columns"].append({"name": "TransactionAmt", "dtype": "float"})
        transaction_value["TransactionAmt"] = transaction_df["TransactionAmt"].values
        transaction_dict["columns"].append({"name": "distance", "dtype": "float"})
        transaction_value["distance"] = transaction_df[["dist1", "dist2"]].values
        counting_features = [f"payment_card_related_counting_{i}" for i in range(1, 15)]
        timedelta_features = [f"timedelta_{i}" for i in range(1, 16)]
        counting_features = transaction_df[counting_features].values 
        timedelta_features = transaction_df[timedelta_features].values
        transaction_dict["columns"].append({"name": "payment_card_related_counting", "dtype": "float" })
        transaction_value["payment_card_related_counting"] = counting_features
        transaction_dict["columns"].append({"name": "timedelta", "dtype": "float" })
        transaction_value["timedelta"] = timedelta_features
        vesta_features = [f"V{i}" for i in range(1, 340)]
        vesta_features = transaction_df[vesta_features].values
        transaction_dict["columns"].append({"name": "vesta_features", "dtype": "float"})
        transaction_value["vesta_features"] = vesta_features
        meta_info['tables'].append(transaction_dict)
        np.savez_compressed(f"{dataset_path}/ieeecis/old/data/transaction.npz", **transaction_value)
        identity_dict["name"] = "Identity"
        identity_dict["source"] = "data/identity.npz"
        identity_dict["format"] = "numpy"
        identity_dict["columns"] = []
        for c in identity_categorical_columns:
            col = {}
            col["name"] = c
            col["dtype"] = "category"
            identity_dict["columns"].append(col)
            identity_value[c] = identity_df[c].values
        identity_dict["columns"].append({"name": "TransactionID", "dtype": "foreign_key", "link_to": "Transaction.TransactionID"})
        identity_value["TransactionID"] = identity_df["TransactionID"].values
        id_related_features = [f"identity_{i}_info" for i in range(1, 12)]
        id_related_features = identity_df[id_related_features].values
        identity_dict["columns"].append({"name": "id_related_features", "dtype": "float"})
        identity_value["id_related_features"] = id_related_features
        meta_info['tables'].append(identity_dict)
        np.savez_compressed(f"{dataset_path}/ieeecis/old/data/identity.npz", **identity_value)        
        meta_info['tasks'] = []
        task_dict = {}
        task_dict["name"] = "fraud"
        task_dict['source'] = "fraud/{split}.npz"
        task_dict['format'] = "numpy"
        task_dict['evaluation_metric'] = "auroc"
        task_dict['target_column'] = "isFraud"
        task_dict['target_table'] = "Transaction"
        task_dict['task_type'] = "classification"
        task_dict['columns'] = transaction_dict["columns"]
        train_splits = {}
        val_splits = {}
        test_splits = {}
        splits_id = np.arange(len(transaction_df))
        train_idx, val_idx, test_idx = train_val_test_split(splits_id, stratify=transaction_df["isFraud"].values, train_size=0.2, val_size=0.1, test_size=0.7)
        for key, val in transaction_value.items():
            train_splits[key] = val[train_idx]
            val_splits[key] = val[val_idx]
            test_splits[key] = val[test_idx]
        np.savez_compressed(f"{dataset_path}/ieeecis/old/fraud/train.npz", **train_splits)
        np.savez_compressed(f"{dataset_path}/ieeecis/old/fraud/validation.npz", **val_splits)
        np.savez_compressed(f"{dataset_path}/ieeecis/old/fraud/test.npz", **test_splits)
        meta_info['tasks'].append(task_dict)
        with open(f"{dataset_path}/ieeecis/old/metadata.yaml", "w") as f:
            yaml.dump(meta_info, f)
        information = analyze_dataframes({'Table Transaction': transaction_value, 'Table Identity': identity_value}, k = 5)
        with open(f"{dataset_path}/ieeecis/information.txt", "w") as f:
            f.write(information)
        print("Processing expert ieeecis dataset")
        ## directly copy everything from old to expert
        shutil.copytree(f"{dataset_path}/ieeecis/old/fraud", f"{dataset_path}/ieeecis/expert/fraud", dirs_exist_ok=True)
        shutil.copytree(f"{dataset_path}/ieeecis/old/data", f"{dataset_path}/ieeecis/expert/data", dirs_exist_ok=True)
        ## open transaction
        transaction_npz = np.load(f"{dataset_path}/ieeecis/expert/data/transaction.npz", allow_pickle=True)
        transaction_dict = dict(transaction_npz)
        ndata = {k:v for k, v in transaction_dict.items() if k in ['match_1', 'match_2', 'match_3', 'match_4', 'match_5', 'match_6', 'match_7', 'match_8', 'match_9']}
        ndata = pd.DataFrame(ndata)
        # import ipdb; ipdb.set_trace()
        df_new = ndata.drop_duplicates().reset_index(drop=True).reset_index()
        df_new.rename(columns={'index': f'MatchStatusID'}, inplace=True)
        # import ipdb; ipdb.set_trace()
        new_table = {col: df_new[col].values for col in df_new.columns}
        ndata = ndata.merge(df_new, on=['match_1', 'match_2', 'match_3', 'match_4', 'match_5', 'match_6', 'match_7', 'match_8', 'match_9'], how='left')
        data = {k:v for k, v in transaction_dict.items() if k not in ['match_1', 'match_2', 'match_3', 'match_4', 'match_5', 'match_6', 'match_7', 'match_8', 'match_9']}
        data[f'MatchStatusID'] = ndata['MatchStatusID'].values
        np.savez_compressed(f"{dataset_path}/ieeecis/expert/data/transaction.npz", **data)
        np.savez_compressed(f"{dataset_path}/ieeecis/expert/data/MatchStatus.npz", **new_table)


        expert_yaml = """
dataset_name: ieeecis
tables:
- columns:
  - dtype: foreign_key
    link_to: Product.ProductCode
    name: ProductCode
  - dtype: float
    name: card_meta_info_1
  - dtype: float
    name: card_meta_info_2
  - dtype: float
    name: card_meta_info_3
  - dtype: foreign_key
    link_to: CardNetwork.NetworkType
    name: card_meta_info_4
  - dtype: float
    name: card_meta_info_5
  - dtype: foreign_key
    link_to: CardType.Type
    name: card_meta_info_6
  - dtype: float
    name: purchaser billing region
  - dtype: float
    name: purchaser billing country
  - dtype: foreign_key
    link_to: EmailDomain.Domain
    name: purchaser email domain
  - dtype: foreign_key
    link_to: EmailDomain.Domain
    name: recipient email domain
  - dtype: primary_key
    name: TransactionID
  - dtype: category
    name: isFraud
  - dtype: float
    name: TransactionAmt
  - dtype: float
    name: distance
  - dtype: float
    name: payment_card_related_counting
  - dtype: float
    name: timedelta
  - dtype: float
    name: vesta_features
  - dtype: foreign_key
    link_to: MatchStatus.MatchStatusID
    name: MatchStatusID
  format: numpy
  name: Transaction
  source: data/transaction.npz
  time_column: null
- columns:
  - dtype: category
    name: identity_12_info
  - dtype: float
    name: identity_13_info
  - dtype: float
    name: identity_14_info
  - dtype: category
    name: identity_15_info
  - dtype: category
    name: identity_16_info
  - dtype: float
    name: identity_17_info
  - dtype: float
    name: identity_18_info
  - dtype: float
    name: identity_19_info
  - dtype: float
    name: identity_20_info
  - dtype: float
    name: identity_21_info
  - dtype: float
    name: identity_22_info
  - dtype: category
    name: identity_23_info
  - dtype: float
    name: identity_24_info
  - dtype: float
    name: identity_25_info
  - dtype: float
    name: identity_26_info
  - dtype: category
    name: identity_27_info
  - dtype: category
    name: identity_28_info
  - dtype: category
    name: identity_29_info
  - dtype: category
    name: identity_30_info
  - dtype: category
    name: identity_31_info
  - dtype: float
    name: identity_32_info
  - dtype: category
    name: identity_33_info
  - dtype: category
    name: identity_34_info
  - dtype: category
    name: identity_35_info
  - dtype: category
    name: identity_36_info
  - dtype: category
    name: identity_37_info
  - dtype: category
    name: identity_38_info
  - dtype: category
    name: DeviceType
  - dtype: category
    name: DeviceInfo
  - dtype: foreign_key
    link_to: Transaction.TransactionID
    name: TransactionID
  - dtype: float
    name: id_related_features
  format: numpy
  name: Identity
  source: data/identity.npz
  time_column: null
- columns:
  - dtype: category
    name: match_1
  - dtype: category
    name: match_2
  - dtype: category
    name: match_3
  - dtype: category
    name: match_4
  - dtype: category
    name: match_5
  - dtype: category
    name: match_6
  - dtype: category
    name: match_7
  - dtype: category
    name: match_8
  - dtype: category
    name: match_9
  - dtype: primary_key
    name: MatchStatusID
  format: numpy
  name: MatchStatus
  source: data/MatchStatus.npz
  time_column: null
tasks:
- columns:
  - dtype: foreign_key
    link_to: Product.ProductCode
    name: ProductCode
  - dtype: float
    name: card_meta_info_1
  - dtype: float
    name: card_meta_info_2
  - dtype: float
    name: card_meta_info_3
  - dtype: foreign_key
    link_to: CardNetwork.NetworkType
    name: card_meta_info_4
  - dtype: float
    name: card_meta_info_5
  - dtype: foreign_key
    link_to: CardType.Type
    name: card_meta_info_6
  - dtype: float
    name: purchaser billing region
  - dtype: float
    name: purchaser billing country
  - dtype: foreign_key
    link_to: EmailDomain.Domain
    name: purchaser email domain
  - dtype: foreign_key
    link_to: EmailDomain.Domain
    name: recipient email domain
  - dtype: primary_key
    name: TransactionID
  - dtype: category
    name: isFraud
  - dtype: float
    name: TransactionAmt
  - dtype: float
    name: distance
  - dtype: float
    name: payment_card_related_counting
  - dtype: float
    name: timedelta
  - dtype: float
    name: vesta_features
  evaluation_metric: auroc
  format: numpy
  key_prediction_label_column: label
  key_prediction_query_idx_column: query_idx
  name: fraud
  source: fraud/{split}.npz
  target_column: isFraud
  target_table: Transaction
  task_type: classification
  time_column: null
"""
        with open(f"{dataset_path}/ieeecis/expert/metadata.yaml", "w") as f:
            f.write(expert_yaml)
    elif dataset == "mvls":
        print("Processing movielens dataset")   
        movies_df = pd.read_csv(f"{dataset_path}/movielens/raw/ml-latest-small/movies.csv")
        ratings_df = pd.read_csv(f"{dataset_path}/movielens/raw/ml-latest-small/ratings.csv")
        tags_df = pd.read_csv(f"{dataset_path}/movielens/raw/ml-latest-small/tags.csv")
        meta_info = {}
        meta_info['dataset_name'] = "movielens"
        meta_info['tables'] = []
        movies_table = {}
        ratings_table = {}
        tags_table = {}
        genres_table = {}
        movie_values = {}
        ratings_values = {}
        tags_values = {}
        movie_values["movieID"] = movies_df["movieId"].values
        movie_values["title"] = movies_df["title"].values
        original_genres = movies_df["genres"].values
        genres = [x.split("|") for x in original_genres]
        movie_values["genres"] = genres
        movie_values = pd.DataFrame.from_dict(movie_values)
        new_df = movie_values[['movieID', 'genres']].explode('genres').reset_index(drop=True)
        new_df = new_df.rename(columns={'genres': 'genre'})
        new_file_path = f"{dataset_path}/movielens/expert/data/genres.pqt"
        new_df.to_parquet(new_file_path)
        movies_table["name"] = "Movies"
        movies_table["source"] = "data/movies.pqt"
        movies_table["format"] = "parquet"
        movies_table["columns"] = []
        movies_table["columns"].append({"name": "movieID", "dtype": "primary_key"})
        movies_table["columns"].append({"name": "title", "dtype": "text"})
        genres_table["name"] = "Genres"
        genres_table["source"] = "data/genres.pqt"
        genres_table["format"] = "parquet"
        genres_table["columns"] = []
        genres_table["columns"].append({"name": "movieID", "dtype": "foreign_key", "link_to": "Movies.movieID"})
        genres_table["columns"].append({"name": "genre", "dtype": "foreign_key", "link_to": "Genres.genre"})
        movie_values = movie_values.drop(columns=["genres"])
        movie_values.to_parquet(f"{dataset_path}/movielens/expert/data/movies.pqt")
        ratings_values["userID"] = ratings_df["userId"].values
        ratings_values["movieID"] = ratings_df["movieId"].values
        ratings_values["rating"] = ratings_df["rating"].values
        timestamp_value = ratings_df["timestamp"].values
        ratings_datetime = np.array([datetime.datetime.fromtimestamp(x) for x in timestamp_value])
        ratings_values["timestamp"] = ratings_datetime
        ratings_values = pd.DataFrame.from_dict(ratings_values)
        ratings_values.to_parquet(f"{dataset_path}/movielens/expert/data/ratings.pqt")
        ratings_table["name"] = "Ratings"
        ratings_table["source"] = "data/ratings.pqt"
        ratings_table["format"] = "parquet"
        ratings_table["columns"] = []
        ratings_table["columns"].append({"name": "userID", "dtype": "foreign_key", "link_to": "Users.userID"})
        ratings_table["columns"].append({"name": "movieID", "dtype": "foreign_key", "link_to": "Movies.movieID"})
        ratings_table["columns"].append({"name": "rating", "dtype": "category"})
        ratings_table["columns"].append({"name": "timestamp", "dtype": "datetime"})
        ratings_table["time_column"] = "timestamp"
        meta_info['tables'].append(movies_table)
        meta_info['tables'].append(ratings_table)
        meta_info['tables'].append(genres_table)
        tags_table["name"] = "Tags"
        tags_table["source"] = "data/tags.pqt"
        tags_table["format"] = "parquet"
        tags_table["columns"] = []
        tags_table["columns"].append({"name": "userID", "dtype": "foreign_key", "link_to": "Users.userID"})
        tags_table["columns"].append({"name": "movieID", "dtype": "foreign_key", "link_to": "Movies.movieID"})
        tags_table["columns"].append({"name": "tag", "dtype": "foreign_key", "link_to": "Tags.tag"})    
        tags_table["columns"].append({"name": "timestamp", "dtype": "datetime"})
        tags_table["time_column"] = "timestamp"
        meta_info['tables'].append(tags_table)
        tags_values["userID"] = tags_df["userId"].values
        tags_values["movieID"] = tags_df["movieId"].values
        tags_values["tag"] = tags_df["tag"].values
        timestamp_value = tags_df["timestamp"].values
        tags_datetime = np.array([datetime.datetime.fromtimestamp(x) for x in timestamp_value])
        tags_values["timestamp"] = tags_datetime
        tags_values = pd.DataFrame.from_dict(tags_values)
        tags_values.to_parquet(f"{dataset_path}/movielens/expert/data/tags.pqt")
        meta_info['tasks'] = []
        task_dict = {}
        task_dict["name"] = "ratings"
        task_dict['source'] = "ratings/{split}.pqt"
        task_dict['format'] = "parquet"
        task_dict['evaluation_metric'] = "auroc"
        task_dict['target_column'] = "rating"
        task_dict['target_table'] = "Ratings"
        task_dict['task_type'] = "classification"
        task_dict['columns'] = ratings_table["columns"]
        meta_info['tasks'].append(task_dict)
        train_splits, val_splits, test_splits = train_val_test_split_by_temporal(ratings_values, "timestamp", train_ratio=0.6, val_ratio=0.2)
        train_splits.to_parquet(f"{dataset_path}/movielens/expert/ratings/train.pqt")
        val_splits.to_parquet(f"{dataset_path}/movielens/expert/ratings/validation.pqt")
        test_splits.to_parquet(f"{dataset_path}/movielens/expert/ratings/test.pqt")
        with open(f"{dataset_path}/movielens/expert/metadata.yaml", "w") as f:
            yaml.dump(meta_info, f)
        print("Processing raw movielens dataset")   
        movies_df = pd.read_csv(f"{dataset_path}/movielens/raw/ml-latest-small/movies.csv")
        ratings_df = pd.read_csv(f"{dataset_path}/movielens/raw/ml-latest-small/ratings.csv")
        tags_df = pd.read_csv(f"{dataset_path}/movielens/raw/ml-latest-small/tags.csv")
        meta_info = {}
        meta_info['dataset_name'] = "movielens"
        meta_info['tables'] = []
        movies_table = {}
        ratings_table = {}
        tags_table = {}
        movie_values = {}
        ratings_values = {}
        tags_values = {}
        movie_values["movieID"] = movies_df["movieId"].values
        movie_values["title"] = movies_df["title"].values
        original_genres = movies_df["genres"].values
        genres = [x.split("|") for x in original_genres]
        movie_values["genres"] = genres
        movies_table["name"] = "Movies"
        movies_table["source"] = "data/movies.pqt"
        movies_table["format"] = "parquet"
        movies_table["columns"] = []
        movies_table["columns"].append({"name": "movieID", "dtype": "primary_key"})
        movies_table["columns"].append({"name": "title", "dtype": "text"})
        movies_table["columns"].append({"name": "genres", "dtype": "multi_category"})
        movie_values = pd.DataFrame.from_dict(movie_values)
        movie_values.to_parquet(f"{dataset_path}/movielens/old/data/movies.pqt")
        ratings_values['ratingID'] = np.arange(len(ratings_df))
        ratings_values["rate_user"] = ratings_df["userId"].values
        ratings_values["rate_movie"] = ratings_df["movieId"].values
        ratings_values["rating"] = ratings_df["rating"].values
        timestamp_value = ratings_df["timestamp"].values
        ratings_datetime = np.array([datetime.datetime.fromtimestamp(x) for x in timestamp_value])
        ratings_values["timestamp"] = ratings_datetime
        ratings_values = pd.DataFrame.from_dict(ratings_values)
        ratings_values.to_parquet(f"{dataset_path}/movielens/old/data/ratings.pqt")
        ratings_table["name"] = "Ratings"
        ratings_table["source"] = "data/ratings.pqt"
        ratings_table["format"] = "parquet"
        ratings_table["columns"] = []
        ratings_table["columns"].append({"name": "rate_user", "dtype": "category"})
        ratings_table["columns"].append({"name": "rate_movie", "dtype": "foreign_key", "link_to": "Movies.movieID"})
        ratings_table["columns"].append({"name": "rating", "dtype": "category"})
        ratings_table["columns"].append({"name": "timestamp", "dtype": "datetime"})
        ratings_table["columns"].append({"name": "ratingID", "dtype": "primary_key"})
        ratings_table["time_column"] = "timestamp"
        meta_info['tables'].append(movies_table)
        meta_info['tables'].append(ratings_table)
        tags_table["name"] = "Tags"
        tags_table["source"] = "data/tags.pqt"
        tags_table["format"] = "parquet"
        tags_table["columns"] = []
        tags_table["columns"].append({"name": "tag_user", "dtype": "category"})
        tags_table["columns"].append({"name": "tag_movie", "dtype": "foreign_key", "link_to": "Movies.movieID"})
        tags_table["columns"].append({"name": "tag", "dtype": "text"})
        tags_table["columns"].append({"name": "timestamp", "dtype": "datetime"})
        tags_table["time_column"] = "timestamp"
        meta_info['tables'].append(tags_table)
        tags_values["tag_user"] = tags_df["userId"].values
        tags_values["tag_movie"] = tags_df["movieId"].values
        tags_values["tag"] = tags_df["tag"].values
        timestamp_value = tags_df["timestamp"].values
        tags_datetime = np.array([datetime.datetime.fromtimestamp(x) for x in timestamp_value])
        tags_values["timestamp"] = tags_datetime
        tags_values = pd.DataFrame.from_dict(tags_values)
        tags_values.to_parquet(f"{dataset_path}/movielens/old/data/tags.pqt")
        meta_info['tasks'] = []
        task_dict = {}
        task_dict["name"] = "ratings"
        task_dict['source'] = "ratings/{split}.pqt"
        task_dict['format'] = "parquet"
        task_dict['evaluation_metric'] = "auroc"
        task_dict['target_column'] = "rating"
        task_dict['target_table'] = "Ratings"
        task_dict['task_type'] = "classification"
        task_dict['columns'] = ratings_table["columns"]
        meta_info['tasks'].append(task_dict)
        train_splits, val_splits, test_splits = train_val_test_split_by_temporal(ratings_values, "timestamp", train_ratio=0.6, val_ratio=0.2)
        train_splits.to_parquet(f"{dataset_path}/movielens/old/ratings/train.pqt")
        val_splits.to_parquet(f"{dataset_path}/movielens/old/ratings/validation.pqt")
        test_splits.to_parquet(f"{dataset_path}/movielens/old/ratings/test.pqt")
        with open(f"{dataset_path}/movielens/old/metadata.yaml", "w") as f:
            yaml.dump(meta_info, f)
        information = analyze_dataframes({'Table Movie': movie_values, 'Table Ratings': ratings_values, 'Table Tags': tags_values}, k = 5)
        with open(f"{dataset_path}/movielens/information.txt", "w") as f:
            f.write(information) 
            
    elif dataset == "AVS":
        ## generate the expert schema, what we do is a trick offered in kaggle to 
        ## remove redundant data
        transaction_df = pd.read_parquet(f"{dataset_path}/avs/raw/avs/data/transactions.pqt")
        offers_df = pd.read_parquet(f"{dataset_path}/avs/raw/avs/data/offers.pqt")
        offers_unique_cat = offers_df['category'].unique()
        new_transaction_df = transaction_df[transaction_df['category'].isin(offers_unique_cat)]
        history_df = pd.read_parquet(f"{dataset_path}/avs/raw/avs/data/history.pqt")
        new_transaction_df.to_parquet(f"{dataset_path}/avs/expert/data/transactions.pqt")
        offers_df.to_parquet(f"{dataset_path}/avs/expert/data/offers.pqt")
        history_df.to_parquet(f"{dataset_path}/avs/expert/data/history.pqt")
        with open(f"{dataset_path}/avs/raw/avs/metadata.yaml", "r") as f:
            old_yaml = yaml.safe_load(f)
        with open(f"{dataset_path}/avs/expert/metadata.yaml", "w") as f:
            yaml.dump(old_yaml, f)
        shutil.copytree(f"{dataset_path}/avs/raw/avs/repeater", f"{dataset_path}/avs/expert/repeater", dirs_exist_ok=True)
        ## generate the raw schema, basically remove any potential link leakage
        old_yaml = """
dataset_name: avs
tables:
- columns:
  - dtype: category
    name: history_chain
  - dtype: category
    name: market
  - dtype: datetime
    name: offerdate
  - dtype: primary_key
    name: id
  - dtype: category
    name: repeater
  - dtype: foreign_key
    link_to: Offer.offer
    name: offer
  format: parquet
  name: History
  source: data/history.pqt
  time_column: offerdate
- columns:
  - dtype: category
    name: brand
  - dtype: category
    name: offer_category
  - dtype: category
    name: company
  - dtype: float
    name: offervalue
  - dtype: float
    name: quantity
  - dtype: primary_key
    name: offer
  format: parquet
  name: Offer
  source: data/offers.pqt
  time_column: null
- columns:
  - dtype: category
    name: brand
  - dtype: category
    name: trans_category
  - dtype: category
    name: trans_chain
  - dtype: category
    name: trans_company
  - dtype: datetime
    name: date
  - dtype: category
    name: dept
  - dtype: category
    name: productmeasure
  - dtype: float
    name: productsize
  - dtype: float
    name: purchaseamount
  - dtype: float
    name: purchasequantity
  - dtype: foreign_key
    link_to: History.id
    name: id
  format: parquet
  name: Transaction
  source: data/transactions.pqt
  time_column: date
tasks:
- columns:
  - dtype: category
    name: history_chain
  - dtype: category
    name: market
  - dtype: datetime
    name: offerdate
  - dtype: primary_key
    name: id
  - dtype: foreign_key
    link_to: Offer.offer
    name: offer
  - dtype: datetime
    name: timestamp
  - dtype: category
    name: repeater
  evaluation_metric: auroc
  format: parquet
  name: repeater
  source: repeater/{split}.pqt
  target_column: repeater
  target_table: History
  task_type: classification
  time_column: timestamp
""" 
        with open(f"{dataset_path}/avs/old/metadata.yaml", "w") as f:
            f.write(old_yaml)
        new_history_df = history_df.rename(columns={'chain': 'history_chain'})
        new_history_df.to_parquet(f"{dataset_path}/avs/old/data/history.pqt")
        new_offers_df = offers_df.rename(columns={'category': 'offer_category'})
        new_offers_df.to_parquet(f"{dataset_path}/avs/old/data/offers.pqt")
        new_transaction_df = new_transaction_df.rename(columns={'category': 'trans_category', 'chain': 'trans_chain', 'company': 'trans_company'})
        new_transaction_df.to_parquet(f"{dataset_path}/avs/old/data/transactions.pqt")
        new_train_splits = pd.read_parquet(f"{dataset_path}/avs/expert/repeater/train.pqt")
        new_val_splits = pd.read_parquet(f"{dataset_path}/avs/expert/repeater/validation.pqt")
        new_test_splits = pd.read_parquet(f"{dataset_path}/avs/expert/repeater/test.pqt")
        new_train_splits = new_train_splits.rename(columns={'chain': 'history_chain'})
        new_val_splits = new_val_splits.rename(columns={'chain': 'history_chain'})
        new_test_splits = new_test_splits.rename(columns={'chain': 'history_chain'})
        new_train_splits.to_parquet(f"{dataset_path}/avs/old/repeater/train.pqt")
        new_val_splits.to_parquet(f"{dataset_path}/avs/old/repeater/validation.pqt")
        new_test_splits.to_parquet(f"{dataset_path}/avs/old/repeater/test.pqt")
        information = analyze_dataframes({'Table History': new_history_df, 'Table Offers': new_offers_df, 'Table Transaction': new_transaction_df}, k = 5)
        with open(f"{dataset_path}/avs/information.txt", "w") as f:
            f.write(information) 
        
    elif dataset == "RR":
        ## expert directly copy the raw data
        shutil.copytree(f"{dataset_path}/retailrocket/raw/retailrocket", f"{dataset_path}/retailrocket/expert/", dirs_exist_ok=True)
        new_view_df = pd.read_parquet(f"{dataset_path}/retailrocket/expert/data/all_views.pqt")
        new_view_df.to_parquet(f"{dataset_path}/retailrocket/old/data/all_views.pqt")
        new_category_tree_df = pd.read_parquet(f"{dataset_path}/retailrocket/expert/data/category_tree.pqt")
        new_category_tree_df.to_parquet(f"{dataset_path}/retailrocket/old/data/category_tree.pqt")
        new_item_availability_df = pd.read_parquet(f"{dataset_path}/retailrocket/expert/data/item_availability.pqt")
        new_item_availability_df = new_item_availability_df.rename(columns={ 'itemid': 'item_available_itemid'})
        new_item_availability_df.to_parquet(f"{dataset_path}/retailrocket/old/data/item_availability.pqt")
        new_item_category = pd.read_parquet(f"{dataset_path}/retailrocket/expert/data/item_categories.pqt")
        new_item_category['category'] = new_item_category['category'].astype('int')
        new_item_category.to_parquet(f"{dataset_path}/retailrocket/old/data/item_categories.pqt")
        new_item_properties = pd.read_parquet(f"{dataset_path}/retailrocket/expert/data/item_properties.pqt")
        new_item_properties = new_item_properties.rename(columns={'itemid': 'item_property_itemid'})
        new_item_properties['property'] = new_item_properties['property'].astype('int')
        new_item_properties.to_parquet(f"{dataset_path}/retailrocket/old/data/item_properties.pqt")
        new_cvr_train = pd.read_parquet(f"{dataset_path}/retailrocket/expert/cvr/cvr-100k_train.pqt")
        new_cvr_train.to_parquet(f"{dataset_path}/retailrocket/old/cvr/train.pqt")
        new_cvr_val = pd.read_parquet(f"{dataset_path}/retailrocket/expert/cvr/cvr-100k_validation.pqt")
        new_cvr_val.to_parquet(f"{dataset_path}/retailrocket/old/cvr/validation.pqt")
        new_cvr_test = pd.read_parquet(f"{dataset_path}/retailrocket/expert/cvr/cvr-100k_test.pqt")
        new_cvr_test.to_parquet(f"{dataset_path}/retailrocket/old/cvr/test.pqt")
        old_yaml = """
dataset_name: retailrocket
tables:
- columns:
  - dtype: foreign_key
    link_to: Item.itemid
    name: itemid
  - dtype: category
    name: visitorid
  - dtype: category
    name: added_to_cart
  - dtype: datetime
    name: timestamp
  format: parquet
  name: View
  source: data/all_views.pqt
  time_column: timestamp
- columns:
  - dtype: primary_key
    name: categoryid
  - dtype: foreign_key
    link_to: Category.categoryid
    name: parentid
  format: parquet
  name: Category
  source: data/category_tree.pqt
- columns:
  - dtype: category
    name: item_available_itemid
  - dtype: float
    name: available
  - dtype: datetime
    name: timestamp
  format: parquet
  name: ItemAvailability
  source: data/item_availability.pqt
  time_column: timestamp
- columns:
  - dtype: category
    name: itemid
  - dtype: foreign_key
    link_to: Category.categoryid
    name: category
  - dtype: datetime
    name: timestamp
  format: parquet
  name: ItemCategory
  source: data/item_categories.pqt
  time_column: timestamp
- columns:
  - dtype: category
    name: item_property_itemid
  - dtype: category
    name: property
  - dtype: text
    name: value
  - dtype: datetime
    name: timestamp
  format: parquet
  name: ItemProperty
  source: data/item_properties.pqt
  time_column: timestamp
tasks:
- columns:
  - dtype: foreign_key
    link_to: Item.itemid
    name: itemid
  - dtype: category
    name: visitorid
  - dtype: category
    name: added_to_cart
  - dtype: datetime
    name: timestamp
  evaluation_metric: auroc
  format: parquet
  name: cvr
  source: cvr/{split}.pqt
  target_column: added_to_cart
  target_table: View
  task_type: classification
  time_column: timestamp
"""
        with open(f"{dataset_path}/retailrocket/old/metadata.yaml", "w") as f:
            f.write(old_yaml)
        information = analyze_dataframes({'Table View': new_view_df, 'Table Category': new_category_tree_df, 'Table ItemAvailability': new_item_availability_df, 'Table ItemCategory': new_item_category, 'Table ItemProperty': new_item_properties}, k = 5)
        with open(f"{dataset_path}/retailrocket/information.txt", "w") as f:
            f.write(information)
        ## real old, this dataset is special since without introducing this primary key it cannot run
        old_yaml = """
dataset_name: retailrocket
tables:
- columns:
  - dtype: foreign_key
    link_to: Item.itemid
    name: itemid
  - dtype: category
    name: visitorid
  - dtype: category
    name: added_to_cart
  - dtype: datetime
    name: timestamp
  - dtype: primary_key
    name: cvr_id
  format: parquet
  name: View
  source: data/all_views.pqt
  time_column: timestamp
- columns:
  - dtype: primary_key
    name: categoryid
  - dtype: foreign_key
    link_to: Category.categoryid
    name: parentid
  format: parquet
  name: Category
  source: data/category_tree.pqt
- columns:
  - dtype: category
    name: item_available_itemid
  - dtype: float
    name: available
  - dtype: datetime
    name: timestamp
  format: parquet
  name: ItemAvailability
  source: data/item_availability.pqt
  time_column: timestamp
- columns:
  - dtype: category
    name: itemid
  - dtype: foreign_key
    link_to: Category.categoryid
    name: category
  - dtype: datetime
    name: timestamp
  format: parquet
  name: ItemCategory
  source: data/item_categories.pqt
  time_column: timestamp
- columns:
  - dtype: category
    name: item_property_itemid
  - dtype: category
    name: property
  - dtype: text
    name: value
  - dtype: datetime
    name: timestamp
  format: parquet
  name: ItemProperty
  source: data/item_properties.pqt
  time_column: timestamp
tasks:
- columns:
  - dtype: primary_key
    name: cvr_id
  - dtype: foreign_key
    link_to: Item.itemid
    name: itemid
  - dtype: category
    name: visitorid
  - dtype: category
    name: added_to_cart
  - dtype: datetime
    name: timestamp
  evaluation_metric: auroc
  format: parquet
  name: cvr
  source: cvr/{split}.pqt
  target_column: added_to_cart
  target_table: View
  task_type: classification
  time_column: timestamp
"""
        new_view_df['cvr_id'] = np.arange(len(new_view_df))
        new_view_df.to_parquet(f"{dataset_path}/retailrocket/realold/data/all_views.pqt")
        with open(f"{dataset_path}/retailrocket/realold/metadata.yaml", "w") as f:
            f.write(old_yaml)
        new_category_tree_df.to_parquet(f"{dataset_path}/retailrocket/realold/data/category_tree.pqt")
        new_item_availability_df.to_parquet(f"{dataset_path}/retailrocket/realold/data/item_availability.pqt")
        new_item_category.to_parquet(f"{dataset_path}/retailrocket/realold/data/item_categories.pqt")
        new_item_properties.to_parquet(f"{dataset_path}/retailrocket/realold/data/item_properties.pqt")
        cvr_df = new_view_df.sample(n=100000)
        train_splits, val_splits, test_splits = train_val_test_split_by_temporal(cvr_df, "timestamp", train_ratio=0.8, val_ratio=0.1)
        train_splits.to_parquet(f"{dataset_path}/retailrocket/realold/cvr/train.pqt")
        val_splits.to_parquet(f"{dataset_path}/retailrocket/realold/cvr/validation.pqt")
        test_splits.to_parquet(f"{dataset_path}/retailrocket/realold/cvr/test.pqt")

    elif dataset == "diginetica":        
        ## copy expert schema directly from raw
        if not os.path.exists(f"{dataset_path}/diginetica/expert/metadata.yaml"):
          shutil.copytree(f"{dataset_path}/diginetica/raw/diginetica", f"{dataset_path}/diginetica/expert/", dirs_exist_ok=True)
          ## clear unnecessary data 
          os.system(f"rm -rf {dataset_path}/diginetica/expert/data/*_train.pqt")
          os.system(f"rm -rf {dataset_path}/diginetica/expert/data/*_validation.pqt")
          os.system(f"rm -rf {dataset_path}/diginetica/expert/data/*_test.pqt")
          os.system(f"rm -rf {dataset_path}/diginetica/expert/data/*_val.pqt")
        new_query_search_string_tokens = pd.read_parquet(f"{dataset_path}/diginetica/expert/data/query_searchstring_tokens.pqt")
        new_query_search_string_tokens = new_query_search_string_tokens.rename(columns={'token': 'search_token'})
        new_query_search_string_tokens.to_parquet(f"{dataset_path}/diginetica/old/data/query_searchstring_tokens.pqt")
        new_query = pd.read_parquet(f"{dataset_path}/diginetica/expert/data/queries.pqt")
        new_query = new_query.rename(columns={'userId': 'query_userId', 'sessionId': 'query_sessionId'})
        new_query.to_parquet(f"{dataset_path}/diginetica/old/data/queries.pqt")
        new_click = pd.read_parquet(f"{dataset_path}/diginetica/expert/data/clicks.pqt")
        ## this is a bug in the original diginetica dataset, we need to remove these two items
        new_click.to_parquet(f"{dataset_path}/diginetica/old/data/clicks.pqt")
        new_query_result = pd.read_parquet(f"{dataset_path}/diginetica/expert/data/query_results.pqt")
        new_query_result.to_parquet(f"{dataset_path}/diginetica/old/data/query_results.pqt")
        new_view = pd.read_parquet(f"{dataset_path}/diginetica/expert/data/item_views.pqt")
        new_view = new_view.rename(columns = {'userId': 'view_user', 'sessionId': 'view_session'})
        new_view.to_parquet(f"{dataset_path}/diginetica/old/data/item_views.pqt")
        new_purchase = pd.read_parquet(f"{dataset_path}/diginetica/expert/data/purchases.pqt")
        new_purchase = new_purchase.rename(columns = {'userId': 'purchaser', 'sessionId': 'purchase_session'})
        new_purchase.to_parquet(f"{dataset_path}/diginetica/old/data/purchases.pqt")

        old_product_name_token = pd.read_parquet(f"{dataset_path}/diginetica/expert/data/product_name_token.pqt")
        old_product_name_token.to_parquet(f"{dataset_path}/diginetica/old/data/old_product_name_token.pqt")

        new_product = pd.read_parquet(f"{dataset_path}/diginetica/expert/data/products.pqt")
        new_product = duckdb.query("""
            SELECT np.itemId,
                np.categoryId,
                np.pricelog2,
                array_agg(opt.token) AS name_tokens
            FROM new_product AS np
            LEFT JOIN old_product_name_token AS opt
            ON np.itemId = opt.itemId
            GROUP BY np.itemId, np.categoryId, np.pricelog2                        
        """)
        new_product.to_parquet(f"{dataset_path}/diginetica/old/data/products.pqt")
        new_click_train = pd.read_parquet(f"{dataset_path}/diginetica/expert/ctr/ctr-100k_train.pqt")
        new_click_train.to_parquet(f"{dataset_path}/diginetica/old/ctr/ctr-100k_train.pqt")
        new_click_validation = pd.read_parquet(f"{dataset_path}/diginetica/expert/ctr/ctr-100k_validation.pqt")
        new_click_validation.to_parquet(f"{dataset_path}/diginetica/old/ctr/ctr-100k_validation.pqt")
        new_click_test = pd.read_parquet(f"{dataset_path}/diginetica/expert/ctr/ctr-100k_test.pqt")
        new_click_test.to_parquet(f"{dataset_path}/diginetica/old/ctr/ctr-100k_test.pqt")
        new_click_val = pd.read_parquet(f"{dataset_path}/diginetica/expert/ctr/ctr-100k_val.pqt")
        new_click_val.to_parquet(f"{dataset_path}/diginetica/old/ctr/ctr-100k_val.pqt")
        new_purchase_train = pd.read_parquet(f"{dataset_path}/diginetica/expert/purchase/purchase_train.pqt")
        ## rename like the new_purchase
        new_purchase_train = new_purchase_train.rename(columns={'userId': 'purchaser', 'sessionId': 'purchase_session'})
        new_purchase_train.to_parquet(f"{dataset_path}/diginetica/old/purchase/purchase_train.pqt")
        new_purchase_validation = pd.read_parquet(f"{dataset_path}/diginetica/expert/purchase/purchase_validation.pqt")
        new_purchase_validation = new_purchase_validation.rename(columns={'userId': 'purchaser', 'sessionId': 'purchase_session'})
        new_purchase_validation.to_parquet(f"{dataset_path}/diginetica/old/purchase/purchase_validation.pqt")
        new_purchase_test = pd.read_parquet(f"{dataset_path}/diginetica/expert/purchase/purchase_test.pqt")
        new_purchase_test = new_purchase_test.rename(columns={'userId': 'purchaser', 'sessionId': 'purchase_session'})
        new_purchase_test.to_parquet(f"{dataset_path}/diginetica/old/purchase/purchase_test.pqt")
        new_purchase_val = pd.read_parquet(f"{dataset_path}/diginetica/expert/purchase/purchase_val.pqt")
        new_purchase_val = new_purchase_val.rename(columns={'userId': 'purchaser', 'sessionId': 'purchase_session'})
        new_purchase_val.to_parquet(f"{dataset_path}/diginetica/old/purchase/purchase_val.pqt")
        old_yaml = """
dataset_name: diginetica
tables:
- columns:
  - dtype: foreign_key
    link_to: Query.queryId
    name: queryId
  - dtype: foreign_key
    link_to: Product.itemId
    name: itemId
  - dtype: datetime
    name: timestamp
  format: parquet
  name: QueryResult
  source: data/query_results.pqt
  time_column: timestamp
- columns:
  - dtype: foreign_key
    link_to: Query.queryId
    name: queryId
  - dtype: foreign_key
    link_to: Product.itemId
    name: itemId
  - dtype: datetime
    name: timestamp
  format: parquet
  name: Click
  source: data/clicks.pqt
  time_column: timestamp
- columns:
  - dtype: category
    name: view_session
  - dtype: category
    name: view_user
  - dtype: foreign_key
    link_to: Product.itemId
    name: itemId
  - dtype: datetime
    name: timestamp
  format: parquet
  name: View
  source: data/item_views.pqt
  time_column: timestamp
- columns:
  - dtype: category
    name: purchase_session
  - dtype: category
    name: purchaser
  - dtype: foreign_key
    link_to: Product.itemId
    name: itemId
  - dtype: category
    name: ordernumber
  - dtype: datetime
    name: timestamp
  format: parquet
  name: Purchase
  source: data/purchases.pqt
  time_column: timestamp
- columns:
  - dtype: foreign_key
    link_to: Query.queryId
    name: queryId
  - dtype: category
    name: search_token
  format: parquet
  name: QuerySearchstringToken
  source: data/query_searchstring_tokens.pqt
- columns:
  - dtype: primary_key
    name: queryId
  - dtype: category
    name: query_sessionId
  - dtype: category
    name: query_userId
  - dtype: float
    name: duration
  - dtype: category
    name: categoryId
  - dtype: datetime
    name: timestamp
  format: parquet
  name: Query
  source: data/queries.pqt
- columns:
  - dtype: primary_key
    name: itemId
  - dtype: category
    name: categoryId
  - dtype: float
    name: pricelog2
  - dtype: multi_category
    name: name_tokens
  format: parquet
  name: Product
  source: data/products.pqt
tasks:
- columns:
  - dtype: foreign_key
    link_to: Product.itemId
    name: itemId
  - dtype: foreign_key
    link_to: Query.queryId
    name: queryId
  - dtype: datetime
    name: timestamp
  - dtype: category
    name: clicked
  evaluation_metric: auroc
  format: parquet
  name: ctr
  source: ctr/ctr-100k_{split}.pqt
  target_column: clicked
  target_table: Click
  task_type: classification
  time_column: timestamp
- columns:
  - dtype: foreign_key
    link_to: Product.itemId
    name: itemId
  - dtype: foreign_key
    link_to: Session.sessionId
    name: purchase_session
  - dtype: datetime
    name: timestamp
  evaluation_metric: mrr
  format: parquet
  name: purchase
  source: purchase/purchase_{split}.pqt
  target_column: itemId
  target_table: Purchase
  task_type: retrieval
  time_column: timestamp
"""
        with open(f"{dataset_path}/diginetica/old/metadata.yaml", "w") as f:
            f.write(old_yaml)
        print("calculateing information")
        information = analyze_dataframes({'Table QueryResult': new_query_result, 'Table Click': new_click, 'Table View': new_view, 'Table Purchase': new_purchase, 'Table QuerySearchstringToken': new_query_search_string_tokens, 'Table Query': new_query, 'Table Product': new_product}, k = 5)
        with open(f"{dataset_path}/diginetica/information.txt", "w") as f:
            f.write(information)

    elif dataset == 'outbrain':
        ## expert schema, directly copying everything from raw to expert
        shutil.copytree(f"{dataset_path}/outbrain/raw/outbrain-small", f"{dataset_path}/outbrain/expert/", dirs_exist_ok=True)
        ## summary_of_data change
        new_event_df = pd.read_parquet(f"{dataset_path}/outbrain/expert/data/events.pqt")
        new_event_df = new_event_df.rename(columns={'uuid': 'event_uuid'})
        new_event_df.to_parquet(f"{dataset_path}/outbrain/old/data/events.pqt")
        new_pageview_df = pd.read_parquet(f"{dataset_path}/outbrain/expert/data/page_views.pqt")
        new_pageview_df = new_pageview_df.rename(columns={'document_id': 'pv_document_id'})
        new_pageview_df.to_parquet(f"{dataset_path}/outbrain/old/data/page_views.pqt")
        new_click_df = pd.read_parquet(f"{dataset_path}/outbrain/expert/data/clicks.pqt")
        new_click_df = new_click_df.rename(columns={'display_id': 'cl_display_id', 'ad_id': 'cl_ad_id'})
        new_click_df.to_parquet(f"{dataset_path}/outbrain/old/data/clicks.pqt")
        new_promoted_content_df = pd.read_parquet(f"{dataset_path}/outbrain/expert/data/promoted_content.pqt")
        new_promoted_content_df = new_promoted_content_df.rename(columns={'document_id': 'pc_document_id'})
        new_promoted_content_df.to_parquet(f"{dataset_path}/outbrain/old/data/promoted_content.pqt")
        new_documents_meta_df = pd.read_parquet(f"{dataset_path}/outbrain/expert/data/documents_meta.pqt")
        new_documents_meta_df.to_parquet(f"{dataset_path}/outbrain/old/data/documents_meta.pqt")
        new_documents_topic_df = pd.read_parquet(f"{dataset_path}/outbrain/expert/data/documents_topics.pqt")
        new_documents_topic_df = new_documents_topic_df.rename(columns={'document_id': 'dt_document_id'})
        new_documents_topic_df.to_parquet(f"{dataset_path}/outbrain/old/data/documents_topics.pqt")
        new_documents_category_df = pd.read_parquet(f"{dataset_path}/outbrain/expert/data/documents_categories.pqt")
        new_documents_category_df = new_documents_category_df.rename(columns={'document_id': 'dc_document_id'})
        new_documents_category_df.to_parquet(f"{dataset_path}/outbrain/old/data/documents_categories.pqt")
        new_documents_entity_df = pd.read_parquet(f"{dataset_path}/outbrain/expert/data/documents_entities.pqt")
        new_documents_entity_df = new_documents_entity_df.rename(columns={'document_id': 'de_document_id'})
        new_documents_entity_df.to_parquet(f"{dataset_path}/outbrain/old/data/documents_entities.pqt")
        new_train = pd.read_parquet(f"{dataset_path}/outbrain/expert/ctr/train.pqt")
        new_train = new_train.rename(columns={'display_id': 'cl_display_id', 'ad_id': 'cl_ad_id'})
        new_train.to_parquet(f"{dataset_path}/outbrain/old/ctr/train.pqt")
        new_val = pd.read_parquet(f"{dataset_path}/outbrain/expert/ctr/validation.pqt")
        new_val = new_val.rename(columns={'display_id': 'cl_display_id', 'ad_id': 'cl_ad_id'})
        new_val.to_parquet(f"{dataset_path}/outbrain/old/ctr/validation.pqt")
        new_test = pd.read_parquet(f"{dataset_path}/outbrain/expert/ctr/test.pqt")
        new_test = new_test.rename(columns={'display_id': 'cl_display_id', 'ad_id': 'cl_ad_id'})
        new_test.to_parquet(f"{dataset_path}/outbrain/old/ctr/test.pqt")
        ## raw schema, removing potential link leakage
        old_yaml = """
dataset_name: outbrain-small
tables:
  - name: Event
    source: data/events.pqt
    format: parquet
    columns:
      - name: display_id
        dtype: primary_key
      - name: event_uuid
        dtype: category
      - name: document_id
        dtype: foreign_key
        link_to: DocumentsMeta.document_id
      - name: platform
        dtype: category
      - name: timestamp
        dtype: datetime
      - name: geo_location
        dtype: category
    time_column: timestamp
  - name: Pageview
    source: data/page_views.pqt
    format: parquet
    columns:
      - name: uuid
        dtype: category
      - name: pv_document_id
        dtype: foreign_key
        link_to: DocumentsMeta.document_id
      - name: timestamp
        dtype: datetime
      - name: platform
        dtype: category
      - name: geo_location
        dtype: category
      - name: traffic_source
        dtype: category
    time_column: timestamp
  - name: Click
    source: data/clicks.pqt
    format: parquet
    columns:
      - name: cl_display_id
        dtype: foreign_key
        link_to: Event.display_id
      - name: cl_ad_id
        dtype: foreign_key
        link_to: PromotedContent.ad_id
      - name: clicked
        dtype: category
      - name: timestamp
        dtype: datetime
    time_column: timestamp
  - name: PromotedContent
    source: data/promoted_content.pqt
    format: parquet
    columns:
      - name: ad_id
        dtype: primary_key
      - name: pc_document_id
        dtype: foreign_key
        link_to: DocumentsMeta.document_id
      - name: campaign_id
        dtype: category
      - name: advertiser_id
        dtype: category
  - name: DocumentsMeta
    source: data/documents_meta.pqt
    format: parquet
    columns:
      - name: document_id
        dtype: primary_key
      - name: source_id
        dtype: category
      - name: publisher_id
        dtype: category
      - name: publish_time
        dtype: datetime
    time_column: publish_time
  - name: DocumentsTopic
    source: data/documents_topics.pqt
    format: parquet
    columns:
      - name: dt_document_id
        dtype: foreign_key
        link_to: DocumentsMeta.document_id
      - name: topic_id
        dtype: category
      - name: confidence_level
        dtype: float
  - name: DocumentsCategory
    source: data/documents_categories.pqt
    format: parquet
    columns:
      - name: dc_document_id
        dtype: foreign_key
        link_to: DocumentsMeta.document_id
      - name: category_id
        dtype: category
      - name: confidence_level
        dtype: float
  - name: DocumentsEntity
    source: data/documents_entities.pqt
    format: parquet
    columns:
      - name: de_document_id
        dtype: foreign_key
        link_to: DocumentsMeta.document_id
      - name: entity_id
        dtype: category
      - name: confidence_level
        dtype: float
tasks:
  - name: ctr
    source: ctr/{split}.pqt
    format: parquet
    columns:
      - name: cl_display_id
        dtype: foreign_key
        link_to: Event.display_id
      - name: cl_ad_id
        dtype: foreign_key
        link_to: PromotedContent.ad_id
      - name: clicked
        dtype: category
      - name: timestamp
        dtype: datetime
    time_column: timestamp
    evaluation_metric: auroc
    target_column: clicked
    target_table: Click
    task_type: classification
"""
        with open(f"{dataset_path}/outbrain/old/metadata.yaml", "w") as f:
            f.write(old_yaml)
        information = analyze_dataframes({'Table Event': new_event_df, 'Table Pageview': new_pageview_df, 'Table Click': new_click_df, 'Table PromotedContent': new_promoted_content_df, 'Table DocumentsMeta': new_documents_meta_df, 'Table DocumentsTopic': new_documents_topic_df, 'Table DocumentsCategory': new_documents_category_df, 'Table DocumentsEntity': new_documents_entity_df}, k = 5)
        with open(f"{dataset_path}/outbrain/information.txt", "w") as f:
            f.write(information)

    elif dataset == 'stackexchange':
        shutil.copytree(f"{dataset_path}/stackexchange/raw/stackexchange", f"{dataset_path}/stackexchange/expert/", dirs_exist_ok=True)
        ## baseline schema
        old_yaml = """
dataset_name: stackexchange
tables:
- columns:
  - dtype: primary_key
    name: Id
  - dtype: category
    name: Class
  - dtype: datetime
    name: Date
  - dtype: text
    name: Name
  - dtype: category
    name: TagBased
  - dtype: foreign_key
    link_to: Users.Id
    name: UserId
  format: parquet
  name: Badges
  source: data/badges.pqt
  time_column: Date
- columns:
  - dtype: primary_key
    name: Id
  - dtype: datetime
    name: CreationDate
  - dtype: text
    name: Text
  - dtype: foreign_key
    link_to: Posts.Id
    name: PostId
  - dtype: category
    name: CommentedUserId
  format: parquet
  name: Comments
  source: data/comments.pqt
  time_column: CreationDate
- columns:
  - dtype: primary_key
    name: Id
  - dtype: text
    name: Comment
  - dtype: datetime
    name: CreationDate
  - dtype: category
    name: PostHistoryTypeId
  - dtype: text
    name: Text
  - dtype: foreign_key
    link_to: Posts.Id
    name: PostId
  - dtype: category
    name: UserName
  format: parquet
  name: PostHistory
  source: data/postHistory.pqt
  time_column: CreationDate
- columns:
  - dtype: primary_key
    name: Id
  - dtype: datetime
    name: CreationDate
  - dtype: category
    name: LinkTypeId
  - dtype: foreign_key
    link_to: Posts.Id
    name: PostId
  - dtype: foreign_key
    link_to: Posts.Id
    name: RelatedPostId
  format: parquet
  name: PostLink
  source: data/postLinks.pqt
  time_column: CreationDate
- columns:
  - dtype: foreign_key
    link_to: Posts.Id
    name: PostId
  - dtype: foreign_key
    link_to: Tag.Id
    name: TagId
  format: parquet
  name: PostTag
  source: data/postTags.pqt
  time_column: null
- columns:
  - dtype: primary_key
    name: Id
  - dtype: text
    name: Body
  - dtype: datetime
    name: CreationDate
  - dtype: category
    name: PostTypeId
  - dtype: text
    name: Title
  - dtype: category
    name: AcceptedAnswerId
  - dtype: category
    name: LastEditorUserId
  - dtype: category
    name: OwnerUserId
  - dtype: category
    name: ParentId
  format: parquet
  name: Posts
  source: data/posts.pqt
  time_column: CreationDate
- columns:
  - dtype: primary_key
    name: Id
  - dtype: text
    name: TagName
  - dtype: category
    name: ExcerptPostId
  - dtype: foreign_key
    link_to: Posts.Id
    name: WikiPostId
  format: parquet
  name: Tag
  source: data/tags.pqt
  time_column: null
- columns:
  - dtype: primary_key
    name: Id
  - dtype: text
    name: AboutMe
  - dtype: datetime
    name: CreationDate
  - dtype: category
    name: Location
  format: parquet
  name: Users
  source: data/users.pqt
  time_column: CreationDate
- columns:
  - dtype: primary_key
    name: Id
  - dtype: category
    name: BountyAmount
  - dtype: datetime
    name: CreationDate
  - dtype: category
    name: VoteTypeId
  - dtype: foreign_key
    link_to: Posts.Id
    name: PostId
  - dtype: category
    name: UserName
  format: parquet
  name: Vote
  source: data/votes.pqt
  time_column: CreationDate
tasks:
- columns:
  - dtype: primary_key
    name: Id
  - dtype: datetime
    name: Timestamp
  - dtype: category
    name: Target
  evaluation_metric: auroc
  format: parquet
  key_prediction_label_column: label
  key_prediction_query_idx_column: query_idx
  name: churn
  source: churn/{split}.pqt
  target_column: Target
  target_table: Users
  task_type: classification
  time_column: Timestamp
- columns:
  - dtype: primary_key
    name: Id
  - dtype: datetime
    name: CreationDate
  - dtype: category
    name: Target
  evaluation_metric: auroc
  format: parquet
  key_prediction_label_column: label
  key_prediction_query_idx_column: query_idx
  name: upvote
  source: upvote/{split}.pqt
  target_column: Target
  target_table: Posts
  task_type: classification
  time_column: CreationDate
"""
        badges_table = pd.read_parquet(f"{dataset_path}/stackexchange/expert/data/badges.pqt")
        users_table = pd.read_parquet(f"{dataset_path}/stackexchange/expert/data/users.pqt")
        postHistory = pd.read_parquet(f"{dataset_path}/stackexchange/expert/data/postHistory.pqt")
        postHistory_userid = postHistory['UserId']
        ## turn postHistoryid into username stored in users_table
        userid_to_username = users_table.set_index('Id')['DisplayName'].to_dict()
        postHistory['UserName'] = postHistory_userid.map(userid_to_username)
        postHistory.drop(columns=['UserId'], inplace=True)
        postHistory.to_parquet(f"{dataset_path}/stackexchange/old/data/postHistory.pqt")
        badges_table.to_parquet(f"{dataset_path}/stackexchange/old/data/badges.pqt")
        vote = pd.read_parquet(f"{dataset_path}/stackexchange/expert/data/votes.pqt")
        vote_userid = vote['UserId']
        vote['UserName'] = vote_userid.map(userid_to_username)
        vote.drop(columns=['UserId'], inplace=True)
        vote.to_parquet(f"{dataset_path}/stackexchange/old/data/votes.pqt")
        comments = pd.read_parquet(f"{dataset_path}/stackexchange/expert/data/comments.pqt")
        comments.rename(columns={"UserId": "CommentedUserId"}, inplace=True)
        comments.to_parquet(f"{dataset_path}/stackexchange/old/data/comments.pqt")
        posts = pd.read_parquet(f"{dataset_path}/stackexchange/expert/data/posts.pqt")
        posts.to_parquet(f"{dataset_path}/stackexchange/old/data/posts.pqt")
        postLinks = pd.read_parquet(f"{dataset_path}/stackexchange/expert/data/postLinks.pqt")
        postLinks.to_parquet(f"{dataset_path}/stackexchange/old/data/postLinks.pqt")
        tags = pd.read_parquet(f"{dataset_path}/stackexchange/expert/data/tags.pqt")
        tags.to_parquet(f"{dataset_path}/stackexchange/old/data/tags.pqt")
        postTags = pd.read_parquet(f"{dataset_path}/stackexchange/expert/data/postTags.pqt")
        postTags.to_parquet(f"{dataset_path}/stackexchange/old/data/postTags.pqt")
        users_table.to_parquet(f"{dataset_path}/stackexchange/old/data/users.pqt")
        for split in ['train', 'validation', 'test']:
            churn_splits = pd.read_parquet(f"{dataset_path}/stackexchange/expert/churn/target_churn_{split}.pqt")
            upvote_splits = pd.read_parquet(f"{dataset_path}/stackexchange/expert/upvote/target_upvote2_cls_{split}.pqt")

            churn_splits.to_parquet(f"{dataset_path}/stackexchange/old/churn/{split}.pqt")
            upvote_splits.to_parquet(f"{dataset_path}/stackexchange/old/upvote/{split}.pqt")
        with open(f"{dataset_path}/stackexchange/old/metadata.yaml", "w") as f:
            f.write(old_yaml)
        dbb = load_dbb_dataset_from_cfg_path_no_name(f"{dataset_path}/stackexchange/old")
        information = analyze_dataframes({'Table Badges': badges_table, 'Table Users': users_table, 'Table PostHistory': postHistory, 'Table Vote': vote, 'Table Comments': comments, 'Table Posts': posts, 'Table PostLink': postLinks, 'Table Tag': tags, 'Table PostTag': postTags}, k = 5, dbb = dbb)
        with open(f"{dataset_path}/stackexchange/information.txt", "w") as f:
            f.write(information)

if __name__ == '__main__':
    typer.run(main)