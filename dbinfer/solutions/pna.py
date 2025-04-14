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


import logging
from typing import Tuple, Dict, Optional, List, Any, Union
import pydantic
import logging
import copy

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import gml_solution
from .base_gml_solution import BaseGMLSolution, BaseGNN, BaseGNNSolutionConfig
from .graph_dataset_config import GraphConfig
from .gnn import PNAConv, PNAConvConfig, HeteroGNNLayer

logger = logging.getLogger(__name__)
logger.setLevel('DEBUG')

class PNASolutionConfig(BaseGNNSolutionConfig):
    hid_size : int
    dropout : float
    conv : PNAConvConfig = PNAConvConfig()

    # Round up the hidden size to multiples of num_towers here.
    # Otherwise most of num_towers will be forced to 1 in instantiation of
    # EdgePNAConv during hyperparameter search.
    @pydantic.root_validator(skip_on_failure=True)
    def roundup_to_num_heads(cls, data):
        hid_size = data['hid_size']
        num_towers = data['conv'].num_towers
        if hid_size % num_towers != 0:
            new_hid_size = (hid_size // num_towers + 1) * num_towers
            logger.warning(
                f'Cannot divide hid_size ({hid_size}) by num_towers ({num_towers}). '
                f'Rounding up hid_size to {new_hid_size}.'
            )
            data['hid_size'] = new_hid_size
        return data

class HeteroPNA(nn.Module):
    def __init__(
        self,
        graph_config : GraphConfig,
        solution_config : PNASolutionConfig,
        node_in_size_dict : Dict[str, int],
        edge_in_size_dict : Dict[str, int],
        out_size : Optional[int],
        num_layers : int,
    ):
        super().__init__()
        if out_size is None:
            out_size = solution_config.hid_size
        self.out_size = out_size

        assert num_layers > 0
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                cur_layer_in_size = node_in_size_dict
            else:
                cur_layer_in_size = {
                    ntype : solution_config.hid_size
                    for ntype in graph_config.ntypes
                }
            if i == num_layers - 1:
                cur_layer_out_size = out_size
            else:
                cur_layer_out_size = solution_config.hid_size
            self.layers.append(
                HeteroGNNLayer(
                    graph_config,
                    cur_layer_in_size,
                    cur_layer_out_size,
                    edge_in_size_dict,
                    PNAConv,
                    solution_config.conv
                )
            )
        self.dropout = nn.Dropout(solution_config.dropout)

    def forward(
        self,
        mfgs,
        X_node_dict : Dict[str, torch.Tensor],
        X_edge_dicts : List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        # assert len(mfgs) == len(self.layers)
        H_node_dict = X_node_dict
        for i, (layer, mfg, X_edge_dict) in enumerate(zip(self.layers, mfgs, X_edge_dicts)):
            H_node_dict = layer(mfg, H_node_dict, X_edge_dict)
            if i != len(self.layers) - 1:
                H_node_dict = {
                    ntype : self.dropout(F.relu(H))
                    for ntype, H in H_node_dict.items()
                }
        return H_node_dict

class PNA(BaseGNN):

    def create_gnn(
        self,
        node_feat_size_dict : Dict[str, int],
        edge_feat_size_dict : Dict[str, int],
        seed_feat_size : int,
        out_size : Optional[int],
    ) -> nn.Module:
        gnn = HeteroPNA(
            self.data_config.graph,
            self.solution_config,
            node_feat_size_dict,
            edge_feat_size_dict,
            out_size,
            num_layers=len(self.solution_config.fanouts),
        )
        return gnn

@gml_solution
class PNASolution(BaseGMLSolution):

    config_class = PNASolutionConfig
    name = "pna"

    def create_model(self):
        return PNA(self.solution_config, self.data_config)
