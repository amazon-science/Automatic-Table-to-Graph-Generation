import torch
import pandas as pd
import numpy as np
import os.path as osp
from .misc import count_files_in_directory
from sklearn.model_selection import train_test_split

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

"""utils function"""
def apply_model(model, x_num, x_cat=None):
    return model(x_num, x_cat)


@torch.inference_mode()
def evaluate(parts, model, dataloaders, dataset):
    model.eval()
    predictions = {}
    for part in parts:
        assert part in ['train', 'val', 'test']
        predictions[part] = []
        for batch in dataloaders[part]:
            x_num, x_cat, y = (
                (batch[0], None, batch[1])
                if len(batch) == 2
                else batch
            )
            predictions[part].append(apply_model(model, x_num, x_cat))
        predictions[part] = torch.cat(predictions[part]).cpu().numpy()
    prediction_type = None if dataset.is_regression else 'logits'
    return dataset.calculate_metrics(predictions, prediction_type)


def train_val_test_split(indices, stratify=None, train_size=0.8, val_size=0.1, test_size=0.1, random_state=None):
    """
    Splits indices into train, validation, and test sets.

    Args:
        indices (np.ndarray): Array of indices to split.
        stratify (np.ndarray, optional): Array of labels for stratified sampling.
        train_size (float, optional): Proportion of data for the training set.
        val_size (float, optional): Proportion of data for the validation set.
        test_size (float, optional): Proportion of data for the test set.
        random_state (int, optional): Seed for random number generator for reproducibility.

    Returns:
        tuple: Train, validation, and test indices (np.ndarray).
    """
    if train_size + val_size + test_size != 1.0:
        raise ValueError("train_size + val_size + test_size must equal 1.0")

    # Split train and remaining (val + test)
    train_indices, remaining_indices = train_test_split(
        indices, train_size=train_size, stratify=stratify, random_state=random_state
    )

    # Split remaining into val and test
    relative_val_size = val_size / (val_size + test_size)  # Adjust val_size relative to remaining
    val_indices, test_indices = train_test_split(
        remaining_indices, train_size=relative_val_size, stratify=stratify[remaining_indices] if stratify is not None else None, random_state=random_state
    )

    return train_indices, val_indices, test_indices


def find_element_indices(array, subset):
    """Finds the indices of elements in 'array' that are present in 'subset'.

    Args:
        array: The main NumPy array.
        subset: A NumPy array containing the elements to search for.

    Returns:
        A NumPy array containing the indices of the matching elements.
    """

    mask = np.isin(array, subset)  # Create a boolean mask where elements match
    indices = np.where(mask)[0]    # Get indices where the mask is True
    return indices


def extract_train_val_test_id_from_objects(splits_file_path, target_column, original_table, train_ratio = .6, val_ratio = .2, test_ratio = .2, stratify = None):
    num_of_files = count_files_in_directory(splits_file_path)
    if num_of_files == 0 or stratify is not None:
        format = "custom"
    ## determine the type of the splits_file, either a parquet or a numpy npz file
    else:
        if osp.exists(osp.join(splits_file_path, "train.pqt")):
            format = "pqt"
        elif osp.exists(osp.join(splits_file_path, "train.npz")):
            format = "npz"
        else:
            raise ValueError("Splits file must be either a parquet or a numpy npz file")
    if format == 'custom':
        indices = np.arange(len(original_table[target_column]))
        train_indices, val_indices, test_indices = train_val_test_split(indices, stratify, train_ratio, val_ratio, test_ratio)
        splits_dict = {
            'train': train_indices,
            'validation': val_indices,
            'test': test_indices
        }
    # import ipdb; ipdb.set_trace()
    for splits in ['train', 'validation', 'test']:
        full_split_path = osp.join(splits_file_path, f"{splits}.{format}")
        if format == "custom":
            idx_column = splits_dict[splits]
            if isinstance(original_table, pd.DataFrame):
                original_table = original_table.to_dict()
                extracted = original_table.iloc[idx_column]
                full_split_path = osp.join(splits_file_path, f"{splits}.pqt")
                extracted.to_parquet(full_split_path)
            else:
                extracted = {k: v[idx_column] for k, v in original_table.items()}
                full_split_path = osp.join(splits_file_path, f"{splits}.npz")
                np.savez_compressed(full_split_path, **extracted)
            # extracted = original_table.iloc[idx_column]
            
        else:
            if format == "pqt":
                split = pd.read_parquet(full_split_path)
            elif format == "npz":
                split = np.load(full_split_path)
                split = dict(split)
            idx_column = split[target_column]
            if format == 'pqt':
                extracted = original_table.loc[original_table.index.isin(idx_column)]
            else:
                idx_column = find_element_indices(original_table[target_column], idx_column)
                extracted = {k: v[idx_column] for k, v in original_table.items()}
        
            # import ipdb; ipdb.set_trace()
            if format == "npz":
                np.savez_compressed(full_split_path, **extracted) 
            else:
                
                extracted.to_parquet(full_split_path)

        