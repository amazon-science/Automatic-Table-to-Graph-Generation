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


from typing import Tuple, Dict, Optional, List
import pydantic
import numpy as np
import logging
from dbinfer_bench import DBBColumnDType

from ...device import DeviceInfo
from .base import (
    ColumnTransform,
    column_transform,
    ColumnData,
    RDBData,
)
from tqdm import tqdm
import torch
import torch.multiprocessing as mp
import sentence_transformers as ST

logger = logging.getLogger(__name__)
logger.setLevel('DEBUG')



class SbertEmbeddingTransformConfig(pydantic.BaseModel):
    model_name : str = "all-MiniLM-L6-v2"
    dim : int = 384
    # max_num_procs : int = 16
    cache_folder: Optional[str] = "/localscratch/chenzh85/models"
    device = 'cuda:0'
    batch_size: int = 32

def sbert_encode(model, texts, batch_size):
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True,
                convert_to_numpy=True).astype('float32')
    return embeddings

@column_transform
class SbertTextEmbeddingTransform(ColumnTransform):
    config_class = SbertEmbeddingTransformConfig
    name = "sbert_text_embedding"
    input_dtype = DBBColumnDType.text_t
    output_dtypes = [DBBColumnDType.float_t]
    output_name_formatters : List[str] = ["{name}"]

    def __init__(self, config : SbertEmbeddingTransformConfig):
        super().__init__(config)
        self.model = ST.SentenceTransformer(config.model_name)
        assert self.model.vector_size == config.dim, \
            "Dimension of the model does not match the config."

    def fit(
        self,
        column : ColumnData,
        device : DeviceInfo
    ) -> None:
        self.new_meta = {
            'dtype' : self.output_dtypes[0],
            'in_size' : self.config.dim,
        }

    def transform(
        self,
        column : ColumnData,
        device : DeviceInfo
    ) -> List[ColumnData]:
        data = column.data
        new_data = sbert_encode(self.model, data, self.config.batch_size)
        return [ColumnData(self.new_meta, new_data)]
