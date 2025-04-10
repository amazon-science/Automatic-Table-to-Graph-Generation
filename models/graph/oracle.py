"""
    Exploration in terms of linear gnn-based oracle
    not used in this paper
"""
import torch
import os
import numpy as np
import yaml
from dgl.graphbolt.internal import read_edges
from dgl.graphbolt.base import etype_str_to_tuple
from dgl.nn import HeteroEmbedding
import dgl 
from models.graph.models import get_hetero_gnn, LinkPredictor
from tqdm import tqdm
import torch.nn.functional as Fn
import dgl.backend as F
from models.graph.hetero import gen_rel_subset_feature, preprocess_features, \
    extract_relation, read_relation_subsets, generate_gaussian_features
from dgl import remove_edges
from torch.utils.data import DataLoader
from dgl.dataloading.negative_sampler import GlobalUniform
from ogb.linkproppred.evaluate import Evaluator
import typer
from dgl.nn.pytorch.network_emb import MetaPath2Vec
from models.graph.utils import load_pqt_or_npz
from models.graph.utils import construct_graph_for_oracle, construct_dfs_for_oracle
from dbinfer.cli import fit_gml, GMLSolutionChoice, fit_tab, TabMLSolutionChoice
import time
import typer

        

class GOracle:
    def __init__(self, strategy, model = 'rgcn', data_id = 'MAG', 
                 task_name = 'venue', model_config_path = 'configs', ) -> None:
        assert strategy in ['full', 'sample']
        assert model in ['sage', 'dfs']
        self.strategy = strategy
        self.model = model
        self.data_id = data_id
        self.task_name = task_name
        self.model_config_path = model_config_path
        
    
    def get_score(self, saved_path_meta, method = 'r2n'):
        if self.model == 'sage':
            start_time = time.time()
            graph_path = construct_graph_for_oracle(self.data_id, saved_path_meta, method)
            end_time = time.time()
            typer.echo(f"Graph constructed in {end_time - start_time}")
            start_time = time.time()
            val_res, test_res = fit_gml(
                graph_path, 
                self.task_name,
                GMLSolutionChoice('sage'),
                os.path.join(self.model_config_path, self.data_id, f'oracle-{self.task_name}.yaml'),
                None,
                False,
                1,
                False
            )
            end_time = time.time()
            typer.echo(f"Validation score: {val_res}, Test score: {test_res}, Time: {end_time - start_time}")
            return val_res
        elif self.model == 'dfs':
            start_time = time.time()
            graph_path = construct_dfs_for_oracle(self.data_id, saved_path_meta)
            end_time = time.time()
            typer.echo(f"DFS constructed in {end_time - start_time}")
            start_time = time.time()
            val_res, test_res = fit_tab(
                graph_path, 
                self.task_name,
                TabMLSolutionChoice('mlp'),
                os.path.join(self.model_config_path, f'mlp.yaml'),
                None,
                1
            )
            end_time = time.time()
            typer.echo(f"Validation score: {val_res}, Test score: {test_res}, Time: {end_time - start_time}")
            return val_res
            
            
            
        
        



def load_disk_file_as_legacy_dgl_full_graph(dataset_dir, graph_data_path, target_table, targets_col, task, task_type = 'node', format='npz', split_id='paperID', test_temporal = -1, 
                                            ts_column = 'timestamp'):
    graph_data = {}
    with open(graph_data_path, 'r') as f:
        graph_data = yaml.safe_load(f)
    graph_data['graph']["nodes"].sort(key=lambda x: x["type"])
    graph_data['graph']["edges"].sort(key=lambda x: x["type"])
    # Construct node_type_offset and node_type_to_id.
    node_type_offset = [0]
    node_type_to_id = {}
    for ntype_id, node_info in enumerate(graph_data['graph']["nodes"]):
        node_type_to_id[node_info["type"]] = ntype_id
        node_type_offset.append(node_type_offset[-1] + node_info["num"])
    total_num_nodes = node_type_offset[-1]
    # Construct edge_type_offset, edge_type_to_id and coo_tensor.
    edge_type_offset = [0]
    edge_type_to_id = {}
    graph_dgl = {}
    for etype_id, edge_info in enumerate(graph_data['graph']["edges"]):
        edge_type_to_id[edge_info["type"]] = etype_id
        edge_fmt = edge_info["format"]
        edge_path = edge_info["path"]
        src, dst = read_edges(dataset_dir, edge_fmt, edge_path)
        edge_type_offset.append(edge_type_offset[-1] + len(src))
        src_type, _, dst_type = etype_str_to_tuple(edge_info["type"])
        rel_name = f"{src_type}_{dst_type}"
        rel_trip = (src_type, rel_name, dst_type)
        graph_dgl[rel_trip] = (src, dst)
    g = dgl.heterograph(graph_dgl)
    ## if link, we just construct tasks our selves
    return g
    
    ## make sure the target directory exists
    ## only load train and valid splits

def train_metapath2vec_embedding(g, metapath, window_size = 4, epochs = 3):
    model = MetaPath2Vec(g, metapath, window_size=window_size)
    dataloader = DataLoader(torch.arange(g.num_nodes('user')), batch_size=128,
                        shuffle=True, collate_fn=model.sample)
    optimizer = torch.optim.SparseAdam(model.parameters(), lr=0.025)
    for e in range(epochs):
        for (pos_u, pos_v, neg_v) in dataloader:
            loss = model(pos_u, pos_v, neg_v)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model




def train_emb_node(emb_model, predictor, optimizer, labels, target_node_type, train_idx, epochs = 10, lr = 0.01):
    ## get the embeddings of all nodes
    user_nids = torch.LongTensor(emb_model.local_to_global_nid[target_node_type])[train_idx]
    train_labels = labels[train_idx]
    user_emb = emb_model.node_embed(user_nids)
    idx = torch.arange(len(user_nids))
    dataloader = DataLoader(idx, batch_size=128, shuffle=True)
    total_loss = total_examples = 0 
    for batch in dataloader:
        batch_feats = user_emb[batch]
        res = emb_model(batch_feats)
        res = predictor(res)
        loss = Fn.nll_loss(res, train_labels[batch])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_examples += len(batch)
        total_loss += loss.item() * len(batch)
    return total_loss / total_examples
        
    
def evaluate_emb_node(emb_model, predictor, g, labels, target_node_type, val_idx):
    user_nids = torch.LongTensor(emb_model.local_to_global_nid[target_node_type])[val_idx]
    t_labels = labels[val_idx]
    user_emb = emb_model.node_embed(user_nids)
    idx = torch.arange(len(user_nids))
    dataloader = DataLoader(idx, batch_size=128, shuffle=False, drop_last=False)
    gts = []
    preds = []
    for batch in dataloader:
        batch_feats = user_emb[batch]
        res = emb_model(batch_feats)
        res = predictor(res)
        gts.append(t_labels[batch])
        preds.append(res.argmax(dim=1))
    gts = torch.cat(gts)
    preds = torch.cat(preds)
    acc = (gts == preds).float().mean()
    return acc
# model, predictor, node_emb, input_nodes, g, optimizer, pos_train_edge, feat_node_type, target_edge, target_node_type_1, target_node_type_2, device, batch_size = 4096
def train_emb_link(emb_model, predictor, g, optimizer, target_edge,
                   target_edge_node_type_1, target_edge_node_type_2, pos_train_edge, batch_size = 4096):
    optimizer.zero_grad()
    predictor.train()
    neg_sampler = GlobalUniform(1)
    nid1 = torch.LongTensor(emb_model.local_to_global_nid[target_edge_node_type_1])
    nid2 = torch.LongTensor(emb_model.local_to_global_nid[target_edge_node_type_2])
    total_loss = total_examples = 0
    for perm in tqdm(DataLoader(
        range(pos_train_edge.size(1)), batch_size, shuffle=True
    )):
        edge = pos_train_edge[:, perm]
        batch_nid1 = nid1[edge[0]]
        batch_nid2 = nid2[edge[1]]
        target_emb_1 = emb_model.node_embed(batch_nid1)
        target_emb_2 = emb_model.node_embed(batch_nid2)
        pos_out = predictor(target_emb_1[edge[0]], target_emb_2[edge[1]])
        pos_loss = -torch.log(pos_out + 1e-15).mean()
        edge = list(neg_sampler(g, {target_edge: edge[0]}).values())[0]
        neg_out = predictor(target_emb_1[edge[0]], target_emb_2[edge[1]])
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()
        loss = pos_loss + neg_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
        optimizer.step()
        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples
    return total_loss / total_examples

def get_emb_pred(test_edges, emb_model, predictor, target_edge_node_type_1, target_edge_node_type_2, batch_size = 4096):
    preds = []
    nid1 = torch.LongTensor(emb_model.local_to_global_nid[target_edge_node_type_1])
    nid2 = torch.LongTensor(emb_model.local_to_global_nid[target_edge_node_type_2])   
    for perm in DataLoader(range(test_edges.size(1)), batch_size):
        edge = test_edges[:, perm]
        batch_nid1 = nid1[edge[0]]
        batch_nid2 = nid2[edge[1]]
        target_emb_1 = emb_model.node_embed(batch_nid1)
        target_emb_2 = emb_model.node_embed(batch_nid2)
        preds += [predictor(target_emb_1, target_emb_2).squeeze().cpu()]
    pred = torch.cat(preds, dim=0)
    return pred

def evaluate_emb_link(emb_model, predictor, target_edge_node_type_1, target_edge_node_type_2, 
                      pos_val_edge, neg_val_edge, pos_test_edge, neg_test_edge, batch_size = 4096):
    predictor.eval() 
    pos_val_pred = get_pred(pos_val_edge, emb_model, predictor, target_edge_node_type_1, target_edge_node_type_2, batch_size)
    neg_val_pred = get_pred(neg_val_edge, emb_model, predictor, target_edge_node_type_1, target_edge_node_type_2, batch_size)
    pos_test_pred = get_pred(pos_test_edge, emb_model, predictor, target_edge_node_type_1, target_edge_node_type_2, batch_size)
    neg_test_pred = get_pred(neg_test_edge, emb_model, predictor, target_edge_node_type_1, target_edge_node_type_2, batch_size)
    evaluator = Evaluator(name='ogbl-citation2')
    val_res = evaluate_mrr(evaluator, pos_val_pred, neg_val_pred)
    test_res = evaluate_mrr(evaluator, pos_test_pred, neg_test_pred)
    return val_res['mrr'], test_res['mrr']
    

def remove_cache(cache_path):
    for f in os.listdir(cache_path):
        if f.startswith('cache'):
            os.remove(os.path.join(cache_path, f))

def sample_metapath(g, k = 2):
    metapaths = []
    for i in range(k):
        metapath = []
        for ntype in g.ntypes:
            metapath.append(ntype)
        metapaths.append(metapath)
    return metapaths

def get_model_oracle_score(full_graph, seen, target_node_type = "", target_edge_node_type_1 = "", target_edge_node_type_2 = "", task='node', 
                           train_ratio = 0.6, val_ratio = 0.2, sample_ratio=0.1, 
                           lr = 0.01, weight_decay = 5e-4, model_name = 'rgcn', num_epochs = 100, 
                           device = 'cuda', num_classes = 10, hidden_dim = 128, negative_ratio = 1, batch_size = 4096, 
                           cache_path = '/localscratch/chenzh85/oraclecache', non_changed = [], dataset_name = "mag"):
    ## 1. Step 1: sample the graph and get the small graphs
    ## 2. Step 2: train the model
    ## 3. Step 3: evaluate the model with validation set performance
    ## 4. Step 4: return the oracle score
    if len(seen) == 0:
        for ntype in full_graph.ntypes:
            seen[ntype] = full_graph.nodes(ntype)
    if sample_ratio < 1:
        for node_type, node_id in seen.items():
            rel_cache_file = os.path.join(cache_path, f'cache_{dataset_name}_{task}_{node_type}.pt')
            if os.path.exists(rel_cache_file):
                typer.echo(f"Loading relation cache file {rel_cache_file}")
                seen[node_type] = torch.load(rel_cache_file)
                continue
            num_nodes = len(node_id)
            perm = torch.randperm(num_nodes)
            num_sample = int(num_nodes * sample_ratio)
            seen[node_type] = node_id[perm[:num_sample]]
            torch.save(seen[node_type], rel_cache_file)
    seen_graph = dgl.node_subgraph(full_graph, seen)
    input_feat_size = seen_graph.nodes[target_node_type].data['feat'].shape[1]
    if model_name == 'rsgc':
        for ntype in seen_graph.ntypes:
            if ntype == target_node_type:
                continue
            N = seen_graph.num_nodes(ntype)
            seen_graph.nodes[ntype].data['feat'] = generate_gaussian_features(N, input_feat_size)
    ## accomodate rsgc design
    relation_results = []
    if model_name == 'rsgc':
        typer.echo("RSGC selected, generating relation subsets")
        if task == 'link':
            relation_results = extract_relation(seen_graph, 0.5, 8, None, target_edge_node_type_1)
        elif task == 'node':
            relation_results = extract_relation(seen_graph, 0.5, 8, target_node_type, None)
        rel_subsets = read_relation_subsets(relation_results)
        with torch.no_grad():
            node_emb = gen_rel_subset_feature(seen_graph, rel_subsets, 2, device)
            typer.echo(f"Generated {len(node_emb)} relation subsets")
            node_emb = preprocess_features(seen_graph, rel_subsets, 2, device)
    if task == 'link':
        target_edge_node_type_1 = target_edge_node_type_1[0].upper() + target_edge_node_type_1[1:]
        target_edge_node_type_2 = target_edge_node_type_2[0].upper() + target_edge_node_type_2[1:]
        target_edge_type = f"{target_edge_node_type_1}_{target_edge_node_type_2}"
        input_nodes = {}
        for ntype in seen_graph.ntypes:
            input_nodes[ntype] = seen_graph.nodes(ntype).to(device)
        model_fn = get_hetero_gnn(model_name)
        if model_name == 'rgcn':
            model = model_fn(seen_graph, hidden_dim, hidden_dim)
            node_emb = rel_graph_embed(seen_graph, 128).to(device)
        elif model_name == 'm2v':
            model = train_metapath2vec_embedding(
                seen_graph, []
            )
        predictor = LinkPredictor(128, 128, 1, 2, 0.5).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        pbar = tqdm(range(num_epochs))
        best_val_res = 0
        best_test_res = 0
        n = seen_graph.num_edges(target_edge_type)
        src, dst = seen_graph.edges(etype=target_edge_type)
        src, dst = F.asnumpy(src), F.asnumpy(dst)
        n_train, n_val, n_test = (
            int(n * train_ratio),
            int(n * val_ratio),
            int(n * (1 - train_ratio - val_ratio)),
        )
        idx = np.random.permutation(n)
        train_pos_idx = idx[:n_train]
        val_pos_idx = idx[n_train : n_train + n_val]
        test_pos_idx = idx[n_train + n_val :]
        all_validated_edges = np.concatenate([src[val_pos_idx], dst[val_pos_idx]])
        reverse_all_validated_edges = np.concatenate([dst[val_pos_idx], src[val_pos_idx]])
        all_validated_edges = np.concatenate([all_validated_edges, reverse_all_validated_edges])
        neg_src = np.repeat(src, negative_ratio)
        neg_dst = torch.randint(0, seen_graph.num_nodes(target_edge_node_type_2), (len(src) * negative_ratio,))
        # neg_src, neg_dst = negative_sample(
        #     seen_graph, negative_ratio * (n_val + n_test)
        # )
        neg_n_val, neg_n_test = (
            negative_ratio * n_val,
            negative_ratio * n_test,
        )
        neg_val_src, neg_val_dst = neg_src[:neg_n_val], neg_dst[:neg_n_val]
        neg_test_src, neg_test_dst = (
            neg_src[neg_n_val:neg_n_val+neg_n_test+1],
            neg_dst[neg_n_val:neg_n_val+neg_n_test+1],
        )
        pos_train_edges = torch.stack((
            F.tensor(src[train_pos_idx]),
            F.tensor(dst[train_pos_idx]),
        ))
        pos_val_edges = torch.stack((
            F.tensor(src[val_pos_idx]),
            F.tensor(dst[val_pos_idx]),
        ))
        neg_val_edges = torch.stack((F.tensor(neg_val_src), F.tensor(neg_val_dst)))
        pos_test_edges = torch.stack((
            F.tensor(src[test_pos_idx]),
            F.tensor(dst[test_pos_idx]),
        ))
        neg_test_edges = torch.stack((F.tensor(neg_test_src), F.tensor(neg_test_dst)))
        msg_passing_graph = remove_edges(seen_graph, eids = F.tensor(all_validated_edges), etype = target_edge_type)
        best_val_res = 0
        best_test_res = 0
        for e in pbar:
            loss = link_train(model, predictor, node_emb, input_nodes, msg_passing_graph, optimizer, pos_train_edges, 
                              target_node_type, target_edge_type, target_edge_node_type_1, target_edge_node_type_2, device, batch_size) 
            # pbar.set_description(f"Epoch: {e}, Training Loss: {loss}")
            val_acc, test_acc = link_test(model, predictor, node_emb, input_nodes, 
                                          msg_passing_graph, pos_train_edges, pos_val_edges, neg_val_edges, pos_test_edges, neg_test_edges, 
                                          target_node_type, target_edge_node_type_1, target_edge_node_type_2, device, batch_size) 
            pbar.set_description(f"Epoch: {e}, Training Loss: {loss}, Val metric: {val_acc}, Test metric: {test_acc}")
            if val_acc > best_val_res:
                best_val_res = val_acc
                best_test_res = test_acc
    elif task == 'node':
        target_node_type_name = target_node_type[0].upper() + target_node_type[1:]
        seen_mask = seen_graph.nodes[target_node_type_name].data['label'] != -1
        all_idx = torch.arange(seen_graph.num_nodes(target_node_type_name))
        labels = seen_graph.nodes[target_node_type_name].data['label']
        seen_idx = all_idx[seen_mask]
        perm = torch.randperm(len(seen_idx))
        train_idx = seen_idx[perm[:int(len(perm) * train_ratio)]]
        val_idx = seen_idx[perm[int(len(perm) * train_ratio):int(len(perm) * (train_ratio + val_ratio))]]
        test_idx = seen_idx[perm[int(len(perm) * (train_ratio + val_ratio)):]]
        input_nodes = {}
        for ntype in seen_graph.ntypes:
            input_nodes[ntype] = seen_graph.nodes(ntype).to(device)
        model_fn = get_hetero_gnn(model_name)
        model = model_fn(seen_graph, hidden_dim, num_classes)
        node_emb = rel_graph_embed(seen_graph, 128).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        pbar = tqdm(range(num_epochs))
        best_val_res = 0
        best_test_res = 0
        for e in pbar:
            loss = train(model, seen_graph, node_emb, input_nodes, optimizer, train_idx, labels, target_node_type_name, device) 
            # pbar.set_description(f"Epoch: {e}, Training Loss: {loss}")
            val_acc, test_acc = evaluate(model, node_emb, input_nodes, seen_graph, labels, val_idx, test_idx, target_node_type_name, device)
            pbar.set_description(f"Epoch: {e}, Training Loss: {loss}, Val Acc: {val_acc}, Test Acc: {test_acc}")
            if val_acc > best_val_res:
                best_val_res = val_acc
                best_test_res = test_acc
    return best_test_res    


def rel_graph_embed(graph, embed_size, target_type = 'paper'):
    node_num = {}
    for ntype in graph.ntypes:
        if ntype == target_type:
            continue
        node_num[ntype] = graph.num_nodes(ntype)
    embeds = HeteroEmbedding(node_num, embed_size)
    return embeds


    


def extract_embed(node_embed, input_nodes, target_type = 'paper'):
    emb = node_embed(
        {ntype: input_nodes[ntype] for ntype in input_nodes if ntype != target_type}
    )
    return emb

def link_train(model, predictor, node_emb, input_nodes, g, optimizer, pos_train_edge, feat_node_type, target_edge, target_node_type_1, target_node_type_2, device, batch_size = 4096):
    optimizer.zero_grad()
    model.train()
    predictor.train()
    neg_sampler = GlobalUniform(1)
    total_loss = total_examples = 0
    for perm in tqdm(DataLoader(
        range(pos_train_edge.size(1)), batch_size, shuffle=True
    )):
        feat_node_type = feat_node_type[0].upper() + feat_node_type[1:]
        emb = extract_embed(node_emb, input_nodes, feat_node_type)
        emb.update({feat_node_type: g.ndata['feat'][feat_node_type]})
        emb = {k: v.to(device) for k, v in emb.items()}
        model = model.to(device)
        g = g.to(device)
        h = model(emb, g)
        target_emb_1 = h[target_node_type_1]
        target_emb_2 = h[target_node_type_2]
        edge = pos_train_edge[:, perm]
        pos_out = predictor(target_emb_1[edge[0]], target_emb_2[edge[1]])
        pos_loss = -torch.log(pos_out + 1e-15).mean()
        edge = list(neg_sampler(g, {target_edge: edge[0]}).values())[0]
        neg_out = predictor(target_emb_1[edge[0]], target_emb_2[edge[1]])
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()
        loss = pos_loss + neg_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
        optimizer.step()
        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples
    return total_loss / total_examples

def train(model, g, node_emb, input_nodes, optimizer, train_idx, labels, target_node_type, device, type='rsgc'):
    optimizer.zero_grad()
    model.train()
    if type == 'rgcn':
    # seeds = g.nodes[target_node_type].data['_ID']
        emb = extract_embed(node_emb, input_nodes, target_node_type)
        emb.update({target_node_type: g.ndata['feat'][target_node_type]})
        emb = {k: v.to(device) for k, v in emb.items()}
        lbls = labels[train_idx].to(device)
        model = model.to(device)
        g = g.to(device)
        # import ipdb; ipdb.set_trace()
        logits = model(g = g, h = emb)[target_node_type]
        y_hat = logits.log_softmax(dim=-1)
        loss = Fn.nll_loss(y_hat[train_idx], lbls)
    else:
        dataloader = torch.utils.data.DataLoader(
            train_idx, batch_size=1024, shuffle=True, drop_last=False)
        for batch in dataloader:
            batch_feats = [x[batch].to(device) for x in node_emb] 
            res = model(batch_feats, g)
            loss = Fn.nll_loss(res, labels[batch].to(device))
    loss.backward()
    optimizer.step()
    return loss.item()  
    

@torch.no_grad()
def evaluate(model, node_emb, input_nodes, g, labels, val_idx, test_idx, target_node_type, device, type = 'rsgc'):
    model.eval()
    if type == 'rgcn':
        emb = extract_embed(node_emb, input_nodes, target_node_type)
        emb.update({target_node_type: g.ndata['feat'][target_node_type]})
        emb = {k: v.to(device) for k, v in emb.items()}
        model = model.to(device)
        g = g.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            logits = model(g = g, h = emb)[target_node_type]
            y_hat = logits.log_softmax(dim=-1).argmax(dim=-1)
    else:
        y_hat = []
        idxs = g.num_nodes(target_node_type)
        dataloader = torch.utils.data.DataLoader(
            range(idxs), batch_size=1024, shuffle=False, drop_last=False)
        for batch in dataloader:
            batch_feats = [x[batch].to(device) for x in node_emb] 
            res = model(batch_feats, g)
            s_y_hat = res.argmax(dim=1).cpu()
            y_hat.append(s_y_hat)
        y_hat = torch.cat(y_hat)
    val_acc = (y_hat[val_idx] == labels[val_idx]).float().mean()
    test_acc = (y_hat[test_idx] == labels[test_idx]).float().mean()
    return val_acc, test_acc

def get_pred(test_edges, h1, h2, predictor, batch_size):
        preds = []
        for perm in DataLoader(range(test_edges.size(1)), batch_size):
            edge = test_edges[:, perm]
            preds += [predictor(h1[edge[0]], h2[edge[1]]).squeeze().cpu()]
        pred = torch.cat(preds, dim=0)
        return pred


def evaluate_mrr(evaluator, y_pred_pos, y_pred_neg):
    y_pred_neg = y_pred_neg.view(y_pred_pos.shape[0], -1)
    results = {}
    mrr = (
        evaluator.eval(
            {
                "y_pred_pos": y_pred_pos,
                "y_pred_neg": y_pred_neg,
            }
        )["mrr_list"]
        .mean()
        .item()
    )

    results["mrr"] = mrr

    return results


@torch.no_grad()
def link_test(model, predictor, node_emb, input_nodes, g,
              pos_train_edge, pos_val_edge, neg_val_edge, pos_test_edge, neg_test_edge,
              feat_node_type, target_node_type_1, target_node_type_2, device, batch_size = 4096, k = 1000):
    model.eval()
    predictor.eval()
    feat_node_type = feat_node_type[0].upper() + feat_node_type[1:]
    emb = extract_embed(node_emb, input_nodes, feat_node_type) 
    emb.update({feat_node_type: g.ndata['feat'][feat_node_type]})
    emb = {k: v.to(device) for k, v in emb.items()}
    model = model.to(device)
    g = g.to(device)
    h = model(emb, g)
    target_emb_1 = h[target_node_type_1]
    target_emb_2 = h[target_node_type_2]    
    pos_val_pred = get_pred(pos_val_edge, target_emb_1, target_emb_2, predictor, batch_size)
    neg_val_pred = get_pred(neg_val_edge, target_emb_1, target_emb_2, predictor, batch_size)
    pos_test_pred = get_pred(pos_test_edge, target_emb_1, target_emb_2, predictor, batch_size)
    neg_test_pred = get_pred(neg_test_edge, target_emb_1, target_emb_2, predictor, batch_size)
    evaluator = Evaluator(name='ogbl-citation2')
    val_res = evaluate_mrr(evaluator, pos_val_pred, neg_val_pred)
    test_res = evaluate_mrr(evaluator, pos_test_pred, neg_test_pred)
    return val_res['mrr'], test_res['mrr']
    
    
    
        
        
if __name__ == '__main__':
    dataset_dir = "datasets/outbrain-small/outbrain-small-tabgnn_ctr_metadata-graph-r2n"
    g_path = "datasets/outbrain-small/outbrain-small-tabgnn_ctr_metadata-graph-r2n/metadata.yaml"                           
    target_table = "click"
    target_col = "clicked"
    task = "ctr"
    format = "pqt"
    g, seen = load_disk_file_as_legacy_dgl_full_graph(dataset_dir, g_path, target_table, target_col, task, format=format, split_id='clickID')
    num_classes = g.ndata['label'][target_table[0].upper() + target_table[1:]].max().item() + 1
    oracle_score = get_model_oracle_score(g, seen, target_table, task='node', sample_ratio=0.1, num_classes=num_classes)
