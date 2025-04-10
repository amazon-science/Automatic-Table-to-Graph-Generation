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

logger = logging.getLogger(__name__)
logger.setLevel('DEBUG')

import gensim.downloader as api
from gensim.utils import tokenize
import os
from pathlib import Path
if not os.environ.get('GENSIM_DATA_DIR'):
    os.environ['GENSIM_DATA_DIR'] = "/tmp"
if not os.environ.get('HOME'):
    os.environ['HOME'] = "/tmp"


def cache_embedding_to_path(embeddings: np.ndarray, path: str) -> None:
    """
    Save the embeddings to a file.
    
    Args:
        embeddings: The embeddings to save.
    """
    if os.path.exists(path):
        logger.warning(f"File {path} already exists. Overwriting.")
    torch.save(embeddings, path)

def load_embedding_from_path(path: str) -> np.ndarray:
    """
    Load embeddings from a file.
    """
    if not os.path.exists(path):
        logger.error(f"File {path} does not exist.")
        return None
    return torch.load(path)    


class GloveTextEmbeddingTransformConfig(pydantic.BaseModel):
    model_name : str = "glove-twitter"
    dim : int = 100
    max_num_procs : int = 16

def _run_one_proc(proc_id, model, dim, data):
    embeddings = []
    if proc_id == 0:
        iterator = tqdm(data)
    else:
        iterator = data
    for text in iterator:
        if not isinstance(text, str):
            embed = np.zeros(dim)
        else:
            token_list = list(tokenize(text))
            if len(token_list) == 0:
                embed = np.zeros(dim)
            else:
                embed = model.get_mean_vector(token_list)
        embeddings.append(embed)
    return np.stack(embeddings).astype('float32')

@column_transform
class GloveTextEmbeddingTransform(ColumnTransform):
    config_class = GloveTextEmbeddingTransformConfig
    name = "glove_text_embedding"
    input_dtype = DBBColumnDType.text_t
    output_dtypes = [DBBColumnDType.float_t]
    output_name_formatters : List[str] = ["{name}"]

    def __init__(self, config : GloveTextEmbeddingTransformConfig):
        super().__init__(config)
        self.config = config

    def fit(
        self,
        column : ColumnData,
        device : DeviceInfo
    ) -> None:
        self.new_meta = {
            'dtype' : self.output_dtypes[0],
            'in_size' : self.config.dim,
        }
        table_name = column.metadata.get('table', 'unknown_table')
        column_name = column.metadata.get('name', 'unknown_column')
        model_name = self.config.model_name
        dim = self.config.dim

        # Sanitize names for filename safety
        safe_dataset = table_name.replace('/', '_').replace('\\', '_')
        safe_column = column_name.replace('/', '_').replace('\\', '_')
        model_safe = model_name.replace('/', '_').replace('\\', '_')
        cache_dir = Path(os.environ['HOME']) / ".cache" / "dbinfer_bench" / "glove_embeddings"
        cache_file = cache_dir / f"{safe_dataset}_{safe_column}_{model_safe}_{dim}.npy"
        if cache_file.exists():
            return 
        self.model = api.load(f"{self.config.model_name}-{self.config.dim}")
        assert self.model.vector_size == self.config.dim, \
            "Dimension of the model does not match the config."

    def transform(
        self,
        column : ColumnData,
        device : DeviceInfo
    ) -> List[ColumnData]:
        # import ipdb; ipdb.set_trace()
        # Generate cache key based on dataset name, column name, model, and dimension
        table_name = column.metadata.get('table', 'unknown_table')
        column_name = column.metadata.get('name', 'unknown_column')
        model_name = self.config.model_name
        dim = self.config.dim

        # Sanitize names for filename safety
        safe_dataset = table_name.replace('/', '_').replace('\\', '_')
        safe_column = column_name.replace('/', '_').replace('\\', '_')
        model_safe = model_name.replace('/', '_').replace('\\', '_')

        # Define cache directory and file
        cache_dir = Path(os.environ['HOME']) / ".cache" / "dbinfer_bench" / "glove_embeddings"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{safe_dataset}_{safe_column}_{model_safe}_{dim}.npy"

        if cache_file.exists():
            logger.info(f"Loading cached embeddings from {cache_file}")
            new_data = np.load(cache_file)
        else:
            data = column.data
            num_procs = min(device.cpu_count // 2, self.config.max_num_procs)
            if num_procs > 1:
                logger.info("Spawn workers to generate embeddings using multi-CPU kernels.")
                ctx = mp.get_context('spawn')
                worklist = np.array_split(data, num_procs)
                with ctx.Pool(processes=num_procs) as pool:
                    results = []
                    for proc_id in range(num_procs):
                        rst = pool.apply_async(
                            _run_one_proc,
                            (proc_id, self.model, self.config.dim, worklist[proc_id])
                        )
                        results.append(rst)
                    results = [rst.get() for rst in results]
                new_data = np.concatenate(results, axis=0).astype('float32')
            else:
                new_data = _run_one_proc(0, self.model, self.config.dim, data)
            
            logger.info(f"Saving embeddings to cache at {cache_file}")
            np.save(cache_file, new_data)
        
        return [ColumnData(self.new_meta, new_data)]
