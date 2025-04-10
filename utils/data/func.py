import torch_frame
from pandas.api.types import is_numeric_dtype

def convert_tensor_to_list_arrays(tensor):
    """ Convert Pytorch Tensor to a list of arrays
    
    Since a pandas DataFrame cannot save a 2D numpy array in parquet format, it is necessary to
    convert the tensor (1D or 2D) into a list of lists or a list of array. This converted tensor
    can then be used to build a pandas DataFrame, which can be saved in parquet format. However,
    tensor with a dimension greater than or equal to 3D cannot be processed or saved into parquet
    files.

    Parameters:
    tensor: Pytorch Tensor
        The input Pytorch tensor (1D or 2D) to be converted
    
    Returns:
    list_array: list of numpy arrays
        A list of numpy arrays
    """
    
    np_array = tensor.numpy()
    list_array = [np_array[i] for i in range(len(np_array))]

    return list_array

def generate_col_to_stype(df, target_col, task_category):
    col_to_stype = {}
    target_col = df.columns[-1]
    if "clf" in task_category:
        col_to_stype[target_col] = torch_frame.categorical
    else:
        col_to_stype[target_col] = torch_frame.numerical

    for col in df.columns[:-1]:
        if "num" in task_category:
            # "num" implies all features are numerical.
            col_to_stype[col] = torch_frame.numerical
        elif df[col].dtype == float:
            col_to_stype[col] = torch_frame.numerical
        else:
            # Heuristics to decide stype
            if is_numeric_dtype(df[col].dtype) and df[col].nunique() > 10:
                col_to_stype[col] = torch_frame.numerical
            else:
                col_to_stype[col] = torch_frame.categorical

