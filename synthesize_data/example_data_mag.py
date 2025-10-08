import os.path
import shutil
import pandas as pd
import numpy as np
import uuid
import yaml


def load_raw_table(datapath, source, fmt):
    """ Load table data from npz or parquet. """
    if fmt == "numpy":
        arr = np.load(os.path.join(datapath, source), allow_pickle=True)
        if isinstance(arr, np.lib.npyio.NpzFile):
            tmp = {}
            for k in arr.files:
                if k == 'feat':
                    tmp[k] = list(arr['feat'])
                else:
                    tmp[k] = arr[k]
            return pd.DataFrame(tmp)
        else:
            return pd.DataFrame(arr)
    elif fmt == "parquet":
        return pd.read_parquet(os.path.join(datapath, source))
    else:
        raise ValueError(f"Unsupported format: {fmt}")

def save_synthesized_table(df, outpath, dstfile):
    """ Save synthesized table. """
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.join(outpath, dstfile)), exist_ok=True)

    if dstfile.endswith('npz'):
        data_dict = {col: df[col].values for col in df.columns}
        if 'feat' in df.columns:
            data_dict['feat'] = np.vstack(data_dict['feat'])
        np.savez_compressed(os.path.join(outpath, dstfile), **data_dict)
    elif dstfile.endswith('pqt'):
        df.to_parquet(os.path.join(outpath, dstfile), engine="pyarrow")

    print(f"> Saved synthesized table {os.path.join(outpath, dstfile)}.")

def synthesize_column(col, dtype, n_rows):
    """ Generate synthetic column values based on dtype. """
    if dtype == "float" and col.apply(lambda x: isinstance(x, (list, np.ndarray))).any():
        # find dimension of first valid vector
        first_vec = col[col.apply(lambda x: isinstance(x, (list, np.ndarray)))].iloc[0]
        dim = len(first_vec)

        # compute per-dimension mean/std if possible
        mat = np.stack(col.dropna().apply(lambda x: np.array(x)))
        mu = mat.mean(axis=0)
        sigma = mat.std(axis=0)
        sigma[sigma == 0] = 1.0  # avoid degenerate std

        # synthesize vectors
        new_feats = np.random.normal(mu, sigma, size=(n_rows, dim))
        return pd.Series(list(new_feats), dtype="object")

    elif dtype == "float":
        data = col.dropna().values.astype(float)
        if len(data) == 0:
            return np.zeros(n_rows)
        mu, sigma = np.mean(data), np.std(data)
        synth = np.random.normal(mu, sigma if sigma > 0 else 1, size=n_rows)
        return np.clip(synth, data.min(), data.max()).round(4)

    elif dtype == "category":
        freqs = col.value_counts(normalize=True)
        return np.random.choice(freqs.index, size=n_rows, p=freqs.values)

    elif dtype == "datetime":
        col = pd.to_datetime(col, errors="coerce").dropna()
        if col.empty:
            return pd.date_range("2000-01-01", periods=n_rows, freq="D")
        min_date, max_date = col.min(), col.max()
        delta = (max_date - min_date).days
        random_days = np.random.randint(0, delta + 1, size=n_rows)
        return [min_date + pd.Timedelta(days=int(d)) for d in random_days]

    else:
        return np.random.choice(col.dropna(), size=n_rows)

def create_missing_table(col_name, fk_values, n_rows=100):
    """ Create a synthetic table for missing foreign-key references. """
    if n_rows is None:
        n_rows = len(set(fk_values))
    synthetic_ids = [str(uuid.uuid4()) for _ in range(n_rows)]
    df = pd.DataFrame({col_name: synthetic_ids})
    id_map = {orig: np.random.choice(synthetic_ids) for orig in set(fk_values)}
    return df, id_map

def synthesize_table(df, table_meta, fk_maps, n_rows=None, schema_table_names=None):
    """ Synthesize one table given metadata + fk mappings. """
    if n_rows is None:
        n_rows = len(df)

    synthetic_df = pd.DataFrame(index=range(n_rows))
    id_map = {}

    for col_meta in table_meta["columns"]:
        col = col_meta["name"]
        dtype = col_meta["dtype"]

        if dtype == "primary_key":
            new_ids = [str(uuid.uuid4()) for _ in range(n_rows)]
            synthetic_df[col] = new_ids
            # Map original IDs (cycled if needed)
            orig_ids = np.resize(df[col].values, n_rows)
            id_map = dict(zip(orig_ids, new_ids))

        elif dtype == "foreign_key":
            ref_table, ref_col = col_meta["link_to"].split(".")
            ref_key = (ref_table, ref_col)

            if (ref_table not in schema_table_names) or (ref_key not in fk_maps):
                # If n mapping available, to synthesize fresh IDs
                synthetic_df[col] = [str(uuid.uuid4()) for _ in range(n_rows)]
            else:
                ref_map = fk_maps[ref_key]
                orig_vals = np.resize(df[col].values, n_rows)
                synthetic_df[col] = [
                    ref_map.get(v, np.random.choice(list(ref_map.values())))
                    for v in orig_vals
                ]

        else:
            synthetic_df[col] = synthesize_column(df[col], dtype, n_rows)

    return synthetic_df, id_map

def synthesize_database(datapath, meta_file, size_config=None):
    """ Generate synthetic database from metadata.yaml. """
    schema = yaml.safe_load(open(os.path.join(datapath, meta_file)))
    print("Schema:", schema)
    fk_maps = {}
    synthetic_dfs = {}
    remaining = {t["name"]: t for t in schema["tables"]}

    # Keep looping until all tables are processed
    # while len(remaining) > 0:
    for table_name, table_meta in list(remaining.items()):
        print("Processing table", table_name)
        df = load_raw_table(datapath, table_meta["source"], table_meta["format"])
        n_rows = size_config.get(table_name, len(df)) if size_config else len(df)
        synth_df, id_map = synthesize_table(df, table_meta, fk_maps, n_rows, remaining)
        synthetic_dfs[table_name] = (synth_df, table_meta["source"])

        # Store PK mapping
        for col_meta in table_meta["columns"]:
            if col_meta["dtype"] == "primary_key":
                fk_maps[(table_name, col_meta["name"])] = id_map

    # Handle tasks (with {split})
    for task_meta in schema.get("tasks", []):
        for split in ["train", "validation", "test"]:
            source = task_meta["source"].replace("{split}", split)
            if os.path.exists(os.path.join(datapath, source)):
                df = load_raw_table(datapath, source, task_meta["format"])
                task_name = f"{task_meta['name']}_{split}"
                n_rows = size_config.get(task_name, len(df)) if size_config else len(df)
                synth_df, _ = synthesize_table(df, task_meta, fk_maps, n_rows, remaining)
                synthetic_dfs[task_name] = (synth_df, source)

    return synthetic_dfs

def main_mag():
    size_config = {
        "Paper": 1000,
        "Cites": 5000,
        "HasTopic": 2000,
        "AffiliatedWith": 1500,
        "Writes": 2500,
        "venue_train": 500,
        "venue_validation": 200,
        "venue_test": 200,
        "cite_train": 500,
        "cite_validation": 200,
        "cite_test": 200,
        "year_train": 500,
        "year_validation": 200,
        "year_test": 200
    }

    datapath = "./data/datasets/mag/old"
    synthetic_dfs = synthesize_database(datapath, "metadata.yaml", size_config=size_config)

    outpath = "./data/datasets/mag/synthetic"

    for name, (df, dstfile) in synthetic_dfs.items():
        print(f"\nSynthetic Table: {name} (rows={len(df)})")
        print(df.head(5))
        save_synthesized_table(df, outpath, dstfile)

    # copy metadata.yaml file
    shutil.copyfile(os.path.join(datapath, "metadata.yaml"),
                    os.path.join(outpath, "metadata.yaml"))
    # shutil.copyfile(os.path.join(datapath, "information.txt"),
    #                 os.path.join(outpath, "information.txt"))


if __name__ == '__main__':
    main_mag()
