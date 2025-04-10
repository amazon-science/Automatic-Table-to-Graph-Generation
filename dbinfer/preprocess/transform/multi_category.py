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
from typing import List
import pydantic
import numpy as np
import pandas as pd
import logging
from dbinfer_bench import DBBColumnDType
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from ...device import DeviceInfo
from .base import (
    ColumnTransform,
    column_transform,
    ColumnData
)

logger = logging.getLogger(__name__)
logger.setLevel('DEBUG')

def perform_pca(X, n_components=None, explained_variance_ratio=None):
    """
    Performs PCA on the input data and returns the transformed data, 
    principal components, explained variance, and explained variance ratio.

    Args:
        X: A NumPy ndarray representing the input data.
        n_components: The number of principal components to keep (optional).
        explained_variance_ratio: The minimum proportion of variance to retain (optional).

    Returns:
        X_pca: The transformed data in the PCA space.
        components: The principal components (the directions of greatest variance).
        explained_variance: The amount of variance explained by each principal component.
        explained_variance_ratio: The proportion of variance explained by each principal component.
    """

    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Initialize PCA
    pca = PCA(n_components=n_components, explained_variance_ratio=explained_variance_ratio)

    # Fit and transform
    X_pca = pca.fit_transform(X_scaled)
    return X_pca

def encode_vector(row, categories, category_to_index, unseen_category):
    encoded_data = np.zeros((len(row), len(categories) + 1))
    for i, row in enumerate(row):
        for score in row:
            if score in category_to_index:
                encoded_data[i, category_to_index[score]] = 1
            else:
                encoded_data[i, unseen_category] = 1
    return encoded_data


class MultiCategoryTransformConfig(pydantic.BaseModel):
    pass

@column_transform
class MultiCategoryTransform(ColumnTransform):
    config_class = MultiCategoryTransformConfig
    name = "multi_category"
    input_dtype = DBBColumnDType.multi_category_t
    output_dtypes = [DBBColumnDType.float_t]
    output_name_formatters : List[str] = ["{name}"]

    def __init__(self, config : MultiCategoryTransformConfig):
        super().__init__(config)

    def fit(
        self,
        column : ColumnData,
        device : DeviceInfo
    ) -> None:
        # _, self.categories = pd.factorize(column.data, use_na_sentinel=True)
        # import ipdb; ipdb.set_trace()
        ## replace nan with empty arrays
        
        unique_scores = np.array(list(set([score for sublist in column.data if isinstance(sublist, np.ndarray) for score in sublist ])))
        self.categories = unique_scores
        self.unseen_category = len(self.categories)
        self.category_to_index = {category: i for i, category in enumerate(self.categories)}
    
    
    def transform(
        self,
        column : ColumnData,
        device : DeviceInfo
    ) -> List[ColumnData]:
        # import ipdb; ipdb.set_trace()
        # import ipdb; ipdb.set_trace()
        column.data = np.array([x if isinstance(x, np.ndarray) else np.array([], dtype=object) for x in column.data], dtype=object)
        try: 
            encoded_data = encode_vector(column.data, self.categories, self.category_to_index, self.unseen_category)
            if encoded_data.shape[1] >= 5000:
                logger.info(f"Unable to process such features")
                encoded_data = np.zeros((encoded_data.shape[0], 1))
            elif encoded_data.shape[1] >= 500:
                logger.info(f"PCA on {encoded_data.shape[1]} features")
                encoded_data = perform_pca(encoded_data, n_components=100)
            new_data = encoded_data.astype('float32')
        except Exception as e:
            logger.error(f"Error during transformation: {e}")
            new_data = np.zeros((len(column.data), 1))
            new_data = new_data.astype('float32')
        new_meta = copy.deepcopy(column.metadata)
        new_meta['dtype'] = DBBColumnDType.float_t
        # import ipdb; ipdb.set_trace()
        # new_meta['num_categories'] = len(self.categories) + 1    
        return [ColumnData(new_meta, new_data)]
