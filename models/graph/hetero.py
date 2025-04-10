"""
    Exploration in terms of linear gnn-based oracle
    not used in this paper
"""

import torch.nn as nn
import os
import torch
import typer
import random
import dgl
import dgl.function as fn

def extract_triplets(g, dataset_name, task_type, cache_path = '/tmp'):
    node_type = g.ntypes
    node_offset = [0]
    for ntype in node_type:
        num_nodes = g.number_of_nodes(ntype)
        node_offset.append(num_nodes + node_offset[-1])

    cache_file_name = f'{dataset_name}_{task_type}_triplets.pt'
    node_offset = node_offset[:-1]
    
    cache_file = os.path.join(cache_path, cache_file_name)
    cache_file_pt = torch.load(cache_file)
    for etype in g.etypes:
        stype, _, dtype = g.to_canonical_etype(etype)
        type_name = f'{stype}_{dtype}_{etype}'
        if cache_file_pt.get(type_name) is not None:
            typer.echo(f"Skipping {type_name} for cache hits")
            continue
        src, dst = g.all_edges(etype=etype)
        src = src.numpy() + node_offset[node_type.index(stype)]
        dst = dst.numpy() + node_offset[node_type.index(dtype)]
        cache_file_pt[type_name] = (src, dst)
    torch.save(cache_file_pt, cache_file)
    return cache_file_pt


def extract_relation(g, prob = 0.5, num_subsets = 8, target_node_type = None, 
                     target_edge_node_type1 = None):
    edges = []
    for u, v, e in g.edges:
        edges.append((u, v, e))
    n_edges = len(edges)
    subsets = set()
    while len(subsets) < num_subsets:
        selected = []
        for e in edges:
            if random.random() < prob:
                selected.append(e)
        # retry if no edge is selected
        if len(selected) == 0:
            continue
        sorted(selected)
        subsets.add(tuple(selected))
    res = []
    for relation in subsets:
        etypes = []
        # only save subsets that touches target node type
        target_touched = False
        for u, v, e in relation:
            etypes.append(e)
            if target_node_type != None and (u == target_node_type or v == target_node_type):
                target_touched = True
            if target_edge_node_type1 != None and (u == target_edge_node_type1 or v == target_edge_node_type1):
                target_touched = True
            print(etypes, target_touched and "touched" or "not touched")
            if target_touched:
                res.append(",".join(etypes))
    return res

def read_relation_subsets(res):
    print("Reading Relation Subsets:")
    rel_subsets = []
    for line in res:
        relations = line.strip().split(',')
        rel_subsets.append(relations)
        # print(relations)
    return rel_subsets


def gen_rel_subset_feature(g, rel_subset, R, device):
    """
    Build relation subgraph given relation subset and generate multi-hop
    neighbor-averaged feature on this subgraph
    """
    new_edges = {}
    ntypes = set()
    for etype in rel_subset:
        stype, _, dtype = g.to_canonical_etype(etype)
        src, dst = g.all_edges(etype=etype)
        src = src.numpy()
        dst = dst.numpy()
        new_edges[(stype, etype, dtype)] = (src, dst)
        new_edges[(dtype, etype + "_r", stype)] = (dst, src)
        ntypes.add(stype)
        ntypes.add(dtype)
    new_g = dgl.heterograph(new_edges)

    # set node feature and calc deg
    for ntype in ntypes:
        num_nodes = new_g.number_of_nodes(ntype)
        if num_nodes < g.nodes[ntype].data["feat"].shape[0]:
            new_g.nodes[ntype].data["hop_0"] = g.nodes[ntype].data["feat"][:num_nodes, :]
        else:
            new_g.nodes[ntype].data["hop_0"] = g.nodes[ntype].data["feat"]
        deg = 0
        for etype in new_g.etypes:
            _, _, dtype = new_g.to_canonical_etype(etype)
            if ntype == dtype:
                deg = deg + new_g.in_degrees(etype=etype)
        norm = 1.0 / deg.float()
        norm[torch.isinf(norm)] = 0
        new_g.nodes[ntype].data["norm"] = norm.view(-1, 1).to(device)

    res = []

    # compute k-hop feature
    for hop in range(1, R + 1):
        ntype2feat = {}
        for etype in new_g.etypes:
            stype, _, dtype = new_g.to_canonical_etype(etype)
            new_g[etype].update_all(fn.copy_u(f'hop_{hop-1}', 'm'), fn.sum('m', 'new_feat'))
            new_feat = new_g.nodes[dtype].data.pop("new_feat")
            assert("new_feat" not in new_g.nodes[stype].data)
            if dtype in ntype2feat:
                ntype2feat[dtype] += new_feat
            else:
                ntype2feat[dtype] = new_feat
        for ntype in new_g.ntypes:
            assert ntype in ntype2feat  # because subgraph is not directional
            feat_dict = new_g.nodes[ntype].data
            old_feat = feat_dict.pop(f"hop_{hop-1}")
            if ntype == "paper":
                res.append(old_feat.cpu())
            feat_dict[f"hop_{hop}"] = ntype2feat.pop(ntype).mul_(feat_dict["norm"])

    res.append(new_g.nodes["paper"].data.pop(f"hop_{R}").cpu())
    return res

def preprocess_features(g, rel_subsets, R, device):
    # pre-process heterogeneous graph g to generate neighbor-averaged features
    # for each relation subsets
    num_paper, feat_size = g.nodes["paper"].data["feat"].shape
    new_feats = [torch.zeros(num_paper, len(rel_subsets), feat_size) for _ in range(R + 1)]
    print("Start generating features for each sub-metagraph:")
    for subset_id, subset in enumerate(rel_subsets):
        print(subset)
        feats = gen_rel_subset_feature(g, subset, R, device)
        for i in range(R + 1):
            feat = feats[i]
            new_feats[i][:feat.shape[0], subset_id, :] = feat
        feats = None
    return new_feats

def generate_gaussian_features(N, H):
    """
    Generates random Gaussian features as a torch tensor.

    Args:
        N (int): Number of data points.
        H (int): Number of features (dimensionality).

    Returns:
        torch.Tensor: A tensor of shape (N, H) containing random Gaussian features.
    """
    return torch.randn(N, H)  # Generate a tensor of N x H random values from a standard normal distribution


class FeedForwardNet(nn.Module):
    def __init__(self, in_feats, hidden, out_feats, n_layers, dropout):
        super(FeedForwardNet, self).__init__()
        self.layers = nn.ModuleList()
        self.n_layers = n_layers
        if n_layers == 1:
            self.layers.append(nn.Linear(in_feats, out_feats))
        else:
            self.layers.append(nn.Linear(in_feats, hidden))
            for i in range(n_layers - 2):
                self.layers.append(nn.Linear(hidden, hidden))
            self.layers.append(nn.Linear(hidden, out_feats))
        if self.n_layers > 1:
            self.prelu = nn.PReLU()
            self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight, gain=gain)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        for layer_id, layer in enumerate(self.layers):
            x = layer(x)
            if layer_id < self.n_layers - 1:
                x = self.dropout(self.prelu(x))
        return x


class SIGN(nn.Module):
    def __init__(
        self, in_feats, hidden, out_feats, num_hops, n_layers, dropout, input_drop
    ):
        super(SIGN, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.prelu = nn.PReLU()
        self.inception_ffs = nn.ModuleList()
        self.input_drop = input_drop
        for i in range(num_hops):
            self.inception_ffs.append(
                FeedForwardNet(in_feats, hidden, hidden, n_layers, dropout)
            )
        self.project = FeedForwardNet(
            num_hops * hidden, hidden, out_feats, n_layers, dropout
        )

    def forward(self, feats):
        hidden = []
        for feat, ff in zip(feats, self.inception_ffs):
            if self.input_drop:
                feat = self.dropout(feat)
            hidden.append(ff(feat))
        out = self.project(self.dropout(self.prelu(torch.cat(hidden, dim=-1))))
        return torch.log_softmax(out, dim=-1)


class WeightedAggregator(nn.Module):
    def __init__(self, num_feats, in_feats, num_hops):
        super(WeightedAggregator, self).__init__()
        self.agg_feats = nn.ParameterList()
        for _ in range(num_hops):
            self.agg_feats.append(nn.Parameter(torch.Tensor(num_feats, in_feats)))
            nn.init.xavier_uniform_(self.agg_feats[-1])

    def forward(self, feats):
        new_feats = []
        for feat, weight in zip(feats, self.agg_feats):
            new_feats.append((feat * weight.unsqueeze(0)).sum(dim=1).squeeze())
        return new_feats


class PartialWeightedAggregator(nn.Module):
    def __init__(self, num_feats, in_feats, num_hops, sample_size):
        super(PartialWeightedAggregator, self).__init__()
        self.weight_store = []
        self.agg_feats = nn.ParameterList()
        self.discounts = nn.ParameterList()
        self.num_hops = num_hops
        for _ in range(num_hops):
            self.weight_store.append(torch.Tensor(num_feats, in_feats))
            self.agg_feats.append(nn.Parameter(torch.Tensor(sample_size, in_feats)))
            self.discounts.append(nn.Parameter(torch.Tensor(in_feats)))
            nn.init.xavier_uniform_(self.weight_store[-1])
        self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.num_hops):
            nn.init.zeros_(self.agg_feats[i])
            nn.init.ones_(self.discounts[i])

    def update_selected(self, selected):
        for param, weight, discount in zip(
            self.agg_feats, self.weight_store, self.discounts
        ):
            weight *= discount
            weight[selected] += param.data
        self.reset_parameters()

    def forward(self, args):
        feats, old_sum = args
        new_feats = []
        for feat, weight, old_feat, discount in zip(
            feats, self.agg_feats, old_sum, self.discounts
        ):
            new_feats.append(
                (feat * weight.unsqueeze(0)).sum(dim=1).squeeze() + old_feat * discount
            )
        return new_feats