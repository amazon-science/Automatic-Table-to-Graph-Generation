# Copyright 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.


import copy
from typing import Tuple, Dict, Optional, List
import pydantic
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, QuantileTransformer
import logging
from dbinfer_bench import DBBColumnDType

from ...device import DeviceInfo
from .base import (
    ColumnTransform,
    column_transform,
    ColumnData,
    RDBData,
)

logger = logging.getLogger(__name__)
logger.setLevel('DEBUG')

def fill_numeric_numpy_array(data: np.ndarray) -> np.ndarray:
    """Fill NaN values in a numpy array with a given value.

    Args:
        data: Input numpy array.

    Returns:
        Numpy array with NaN values filled.
    """
    mean_values = np.nanmean(data, axis=0)
    inds = np.where(np.isnan(data))
    data[inds] = np.take(mean_values, inds[1])
    return data

class EmbeddingNaNTransformConfig(pydantic.BaseModel):
    pass

@column_transform
class EmbeddingNaNTransform(ColumnTransform):
    config_class = EmbeddingNaNTransformConfig
    name = "embedding_nan"
    input_dtype = DBBColumnDType.float_t
    output_dtypes = [DBBColumnDType.float_t]
    output_name_formatters : List[str] = ["{name}"]

    def __init__(self, config : EmbeddingNaNTransformConfig):
        super().__init__(config)

    def fit(
        self,
        column : ColumnData,
        device : DeviceInfo
    ) -> None:
        self.new_meta = {
            'dtype' : self.output_dtypes[0],
            'in_size' : 1 if column.data.ndim == 1 else column.data.shape[1]
        }

        
        if column.data.ndim <= 1:
            # Ignore single dimension values.
            return

    def transform(
        self,
        column : ColumnData,
        device : DeviceInfo
    ) -> List[ColumnData]:
        new_data = column.data.astype('float32')
        if column.data.ndim <= 1:
            return [ColumnData(self.new_meta, new_data)]
        scaler = StandardScaler()
        new_data = fill_numeric_numpy_array(new_data)
        new_data = scaler.fit_transform(new_data)
        return [ColumnData(self.new_meta, new_data)]
