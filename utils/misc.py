import random
import os 
import numpy as np
import torch
import enum
from typing import Any, Optional, Union, cast, Tuple, Dict
import scipy.special
import sklearn.metrics as skm
import pickle
import json 
from pathlib import Path
from contextlib import ContextDecorator
from codetiming import Timer
from functools import wraps
from datetime import datetime
import time
from humanfriendly import format_timespan



class TaskType(enum.Enum):
    BINCLASS = 'binclass'
    MULTICLASS = 'multiclass'
    REGRESSION = 'regression'

    def __str__(self) -> str:
        return self.value

def needs_small_lr(name):
    return any(x in name for x in ['.col_head', '.col_tail'])

def raise_unknown(unknown_what: str, unknown_value: Any):
    raise ValueError(f'Unknown {unknown_what}: {unknown_value}')

def seed_everything(seed=42):
    '''
    Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.
    '''
    random.seed(seed)
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # When running on the CuDNN backend, two further options must be set
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class PredictionType(enum.Enum):
    LOGITS = 'logits'
    PROBS = 'probs'


def calculate_rmse(
    y_true: np.ndarray, y_pred: np.ndarray, std: Optional[float]
) -> float:
    rmse = skm.mean_squared_error(y_true, y_pred) ** 0.5
    if std is not None:
        rmse *= std
    return rmse


def _get_labels_and_probs(
    y_pred: np.ndarray, task_type: TaskType, prediction_type: Optional[PredictionType]
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    assert task_type in (TaskType.BINCLASS, TaskType.MULTICLASS)

    if prediction_type is None:
        return y_pred, None

    if prediction_type == PredictionType.LOGITS:
        probs = (
            scipy.special.expit(y_pred)
            if task_type == TaskType.BINCLASS
            else scipy.special.softmax(y_pred, axis=1)
        )
    elif prediction_type == PredictionType.PROBS:
        probs = y_pred
    else:
        raise_unknown('prediction_type', prediction_type)

    assert probs is not None
    labels = np.round(probs) if task_type == TaskType.BINCLASS else probs.argmax(axis=1)
    return labels.astype('int64'), probs


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task_type: Union[str, TaskType],
    prediction_type: Optional[Union[str, PredictionType]],
    y_info: Dict[str, Any],
) -> Dict[str, Any]:
    # Example: calculate_metrics(y_true, y_pred, 'binclass', 'logits', {})
    task_type = TaskType(task_type)
    if prediction_type is not None:
        prediction_type = PredictionType(prediction_type)

    if task_type == TaskType.REGRESSION:
        assert prediction_type is None
        assert 'std' in y_info
        rmse = calculate_rmse(y_true, y_pred, y_info['std'])
        result = {'rmse': rmse}
    else:
        labels, probs = _get_labels_and_probs(y_pred, task_type, prediction_type)
        result = cast(
            Dict[str, Any], skm.classification_report(y_true, labels, output_dict=True)
        )
        if task_type == TaskType.BINCLASS:
            result['roc_auc'] = skm.roc_auc_score(y_true, probs)
    return result

def load_json(path: Union[Path, str], **kwargs) -> Any:
    return json.loads(Path(path).read_text(), **kwargs)


def dump_json(x: Any, path: Union[Path, str], **kwargs) -> None:
    kwargs.setdefault('indent', 4)
    Path(path).write_text(json.dumps(x, **kwargs) + '\n')


def load_pickle(path: Union[Path, str], **kwargs) -> Any:
    return pickle.loads(Path(path).read_bytes(), **kwargs)


def dump_pickle(x: Any, path: Union[Path, str], **kwargs) -> None:
    Path(path).write_bytes(pickle.dumps(x, **kwargs))


def load(path: Union[Path, str], **kwargs) -> Any:
    return globals()[f'load_{Path(path).suffix[1:]}'](Path(path), **kwargs)


def dump(x: Any, path: Union[Path, str], **kwargs) -> Any:
    return globals()[f'dump_{Path(path).suffix[1:]}'](x, Path(path), **kwargs)


def count_files_in_directory(directory_path):
    """
    Counts the number of files in the specified directory.

    Args:
        directory_path (str): The path to the directory.

    Returns:
        int: The number of files found in the directory.
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path, exist_ok=True)
        return 0
    try:
        file_count = len([f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))])
        return file_count
    except FileNotFoundError:
        print(f"Error: Directory not found - '{directory_path}'")
        return None
    except PermissionError:
        print(f"Error: Permission denied for directory - '{directory_path}'")
        return None
    

def seed_everything(seed: int = 42):
    """
    Sets the random seeds for various libraries for reproducibility.

    Args:
        seed: The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If you have a GPU
    
def copy_directory(source_dir, destination_dir):
    """Copies the entire contents of the source directory to the destination, overwriting existing files."""
    for item in os.listdir(source_dir):
        source_item = os.path.join(source_dir, item)
        destination_item = os.path.join(destination_dir)
        os.system("rsync -avz --progress {} {}".format(source_item, destination_item))



def get_cur_time(timezone=None, t_format="%m-%d %H:%M:%S"):
    return datetime.fromtimestamp(int(time.time()), timezone).strftime(t_format)



class timer(ContextDecorator):
    def __init__(self, name=None, log_func=print):
        self.name = name
        self.log_func = log_func
        self.timer = Timer(name=name, logger=None)  # Disable internal logging

    def __enter__(self):
        self.timer.start()
        self.log_func(f"Started {self.name} at {get_cur_time()}")
        return self

    def __exit__(self, *exc):
        elapsed_time = self.timer.stop()
        formatted_time = format_timespan(elapsed_time)
        self.log_func(
            f"Finished {self.name} at {get_cur_time()}, running time = {formatted_time}."
        )
        return False

    def __call__(self, func):
        self.name = self.name or func.__name__

        @wraps(func)
        def decorator(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return decorator
    

def convert_pandas_to_npz_(df):
    """
    Convert a pandas DataFrame to a .npz file.
    
    Args:
    df: The pandas DataFrame to convert.
    
    Returns:
    str: The path to the .npz file.
    """
    npz_file = "temp.npz"
    np.savez(npz_file, df)
    return npz_file

