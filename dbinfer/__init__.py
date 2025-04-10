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


from . import logger
from . import time_budget
from pathlib import Path
from typing import Tuple, Dict, Optional, List, Any

from . import yaml_utils

from dbinfer_bench import DBBRDBDatasetMeta, DBBColumnDType, DBBTaskType, DBBRDBTask, DBBRDBTaskCreator, DBBRDBDatasetCreator
from dbinfer_bench.table_loader import get_table_data_loader
import numpy as np
import sqlalchemy
from sqlalchemy import (
    MetaData,
    Table,
    Column,
    String,
    ForeignKey,
    Uuid,
    Float,
    ARRAY,
    VARCHAR,
    DateTime
)
from dbinfer_bench import download_or_get_path

def _check_dgl_version():
    import dgl
    required_version = "2.1a240205"
    parts = dgl.__version__.split('+')
    current_version = parts[0]
    if current_version != required_version:
        raise RuntimeError(
            f"Required DGL version {required_version} but the installed version is {current_version}."
        )

# _check_dgl_version()

class DBBRDBDataset:

    def __init__(
        self,
        path : Path = None,
        metadata = None,
    ):
        if metadata is None:
            self.full_path = path
            if str(path).endswith('.yaml'):
                path = Path(path).parent
            self.path = Path(path)
            self._metadata = self._load_metadata()
        elif metadata is not None:
            self.path = Path(path)
            if not hasattr(metadata, 'tasks'):
                metadata['tasks'] = []  
            self._metadata = DBBRDBDatasetMeta.parse_obj(metadata)
        else:
            raise ValueError("Either path or metadata should be provided.")
        self._load_data()

    
    def _load_metadata(self):
        if str(self.full_path).endswith('.yaml'):
            return yaml_utils.load_pyd(DBBRDBDatasetMeta, self.full_path)
        elif (self.path / 'metadata.yaml').exists():
            return yaml_utils.load_pyd(DBBRDBDatasetMeta, self.path / 'metadata.yaml')
        else:
            raise ValueError(f"Cannot find metadata.yaml in {self.path}.")

    def _load_data(self):
        # Load tables.
        self._tables = {}
        for table_schema in self.metadata.tables:
            table_path = self.path / table_schema.source
            loader = get_table_data_loader(table_schema.format)
            self._tables[table_schema.name] = loader(table_path)

        # Load tasks.
        self._tasks = []
        for task_meta in self.metadata.tasks:
            loader = get_table_data_loader(task_meta.format)
            def _load_split(split):
                table_path = self.path / task_meta.source.format(split=split)
                return loader(table_path)
            train_set = _load_split('train')
            validation_set = _load_split('validation')
            test_set = _load_split('test')
            self._tasks.append(DBBRDBTask(
                task_meta, train_set, validation_set, test_set))

    @property
    def dataset_name(self) -> str:
        return self.metadata.dataset_name

    @property
    def metadata(self) -> DBBRDBDatasetMeta:
        return self._metadata

    @property
    def tasks(self) -> List[DBBRDBTask]:
        return self._tasks

    @property
    def tables(self) -> Dict[str, Dict[str, np.ndarray]]:
        return self._tables

    def get_task(self, name : str) -> DBBRDBTask:
        for task in self.tasks:
            if task.metadata.name == name:
                return task
        raise ValueError(f"Unknown task {name}.")

    @property
    def sqlalchemy_metadata(self) -> sqlalchemy.MetaData:
        """Get metadata in sqlalchemy structure."""
        metadata = MetaData()
        pks, referred_pks = {}, {}
        for tbl_meta in self.metadata.tables:
            tbl_name = tbl_meta.name
            cols = []
            for col_meta in tbl_meta.columns:
                col_name = col_meta.name
                col_data = self.tables[tbl_name][col_name]
                if col_meta.dtype == DBBColumnDType.float_t:
                    if (col_data.shape) == 1:
                        col = Column(col_name, Float)
                    else:
                        col = Column(col_name, ARRAY(Float))
                elif col_meta.dtype == DBBColumnDType.category_t:
                    col = Column(col_name, VARCHAR)
                elif col_meta.dtype == DBBColumnDType.datetime_t:
                    col = Column(col_name, DateTime)
                elif col_meta.dtype == DBBColumnDType.text_t:
                    col = Column(col_name, String)
                elif col_meta.dtype == DBBColumnDType.foreign_key:
                    col = Column(col_name, None, ForeignKey(col_meta.link_to))
                    link_tbl, link_col = col_meta.link_to.split('.')
                    referred_pks[link_tbl] = link_col
                elif col_meta.dtype == DBBColumnDType.primary_key:
                    col = Column(col_name, Uuid, primary_key=True)
                    pks[tbl_name] = col_name
                else:
                    col = Column(col_name, VARCHAR)
                cols.append(col)
            alchemy_tbl = Table(tbl_name, metadata, *cols)
        # Create missing tables.
        for tbl, col in referred_pks.items():
            if tbl not in pks:
                alchemy_tbl = Table(tbl, metadata, Column(col, Uuid, primary_key=True))
            elif col != pks[tbl]:
                raise ValueError(f"Detect two primary keys ({col} and {pks[tbl]}) for table '{tbl}'!")

        return metadata

    def save(self, path : Path):
        ds_ctor = DBBRDBDatasetCreator(self.metadata.dataset_name)
        ds_ctor.replace_tables_from(self)
        for task in self.tasks:
            task_ctor = DBBRDBTaskCreator(task.metadata.name)
            task_ctor.copy_fields_from(task.metadata)
            task.metadata.column_dict
            for col_name in task.train_set:
                col_meta = dict(task.metadata.column_dict[col_name])
                col_meta.pop('name')
                task_ctor.add_task_data(
                    col_name,
                    task.train_set[col_name],
                    task.validation_set[col_name],
                    task.test_set[col_name],
                    **col_meta)
            if task.metadata.task_type == DBBTaskType.retrieval:
                for col_name in [
                    task.metadata.key_prediction_label_column,
                    task.metadata.key_prediction_query_idx_column,
                ]:
                    task_ctor.add_task_data(
                        col_name,
                        None,
                        task.validation_set[col_name],
                        task.test_set[col_name],
                        dtype=None)
            ds_ctor.add_task(task_ctor)
        ds_ctor.done(path)
        

def load_rdb_data(name_or_path : str) -> DBBRDBDataset:
    path = download_or_get_path(name_or_path)
    return DBBRDBDataset(path)