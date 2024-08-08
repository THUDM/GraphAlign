import os
import torch.multiprocessing
from ogb.nodeproppred import DglNodePropPredDataset
from datasets.dataset_utils import *
import torch
import pandas as pd
import numpy as np
import sys
import json
import torch.nn.functional as F
import dgl
from utils.utils import show_occupied_memory
from sklearn.preprocessing import StandardScaler


def convert_1d_array_to_ego_graph_list(path_or_data, salt=-200000000):
    if type(path_or_data) == str:
        array = np.load(path_or_data)
    else:
        array = path_or_data
    index = np.where(array < 0)[0]
    array[index] -= salt
    ego_graphs = np.split(array, index[1:])
    return ego_graphs

class papers100M_feat_type:
    def __init__(self,feats, node_used_path):
        node_used = np.load(node_used_path)
        node_used = np.sort(node_used)
        self.mapping ={}
        for idx, n in enumerate(node_used):
            self.mapping[n] = idx 
        self.feats = feats

    def __call__(self,index):
        idxs = [self.mapping[idx] for idx in index]
        x = self.feats[idxs]        
        return x.to(torch.float32)

def load_large_dataset(dataset_name, data_dir, ego_graphs_file_path, no_scale, multi_scale, feat_type="e5_float16",drop_model="random"):
    short_name = {"ogbn-papers100M":"100M","ogbn-arxiv":"arxiv","ogbn-products":"products"}
    if dataset_name.startswith("ogbn") and dataset_name!="ogbn-papers100M":
        graph_path = os.path.join(data_dir, dataset_name, f"dgl_graph_{short_name[dataset_name]}_int32") 
        feats_path = os.path.join(data_dir, dataset_name, f"{short_name[dataset_name]}_embedding_{feat_type}.npy")         
        scale_path = os.path.join(data_dir, dataset_name, f"{dataset_name}_stats.pt") 
        ego_graphs_file_path = os.path.join(data_dir, dataset_name, f"{dataset_name}-lc-ego-graphs-256.pt") 
        label_path = os.path.join(data_dir, dataset_name, f'{short_name[dataset_name]}_label.pt')
        split_path = os.path.join(data_dir, dataset_name, f'{short_name[dataset_name]}_split.pt')
    
        graph = dgl.load_graphs(graph_path)[0][0]    
        if "feat" in graph.ndata:
            del graph.ndata["feat"]
        if "year" in graph.ndata:
            del graph.ndata["year"]
        if dataset_name == "ogbn-arxiv" and drop_model== "random":
            graph = dgl.to_bidirected(graph)        
        graph = graph.remove_self_loop().add_self_loop()

        split_idx = torch.load(split_path)
        
        labels = torch.load(label_path)
        
        if "float16" in feat_type:
            feats = torch.tensor(np.load(feats_path),dtype=torch.float16) 
        else:
            feats = torch.tensor(np.load(feats_path),dtype=torch.float32) 
   
        if feat_type=="e5_float16":
            _mean, _std = torch.load(scale_path)
            _mean = _mean.to(torch.float16)

        if no_scale == False:
            if feat_type=="e5_float16":
                feats = feats - _mean
            else: 
                feats = center_feats(feats)
                   
        train_lbls = labels[split_idx["train"]]
        val_lbls = labels[split_idx["valid"]]
        test_lbls = labels[split_idx["test"]]
        labels = torch.cat([train_lbls, val_lbls, test_lbls])

        nodes = torch.load(ego_graphs_file_path) 
        return feats, graph, labels, split_idx, nodes
        
 
    elif dataset_name=="ogbn-papers100M":
        feats_path = os.path.join(data_dir, dataset_name, f"100M_embedding_{feat_type}_used.npy")
        lable_path = os.path.join(data_dir, dataset_name, "100M-node-label.npz")
        used_node_path = os.path.join(data_dir, dataset_name, "used_node.npy")
        scale_path = os.path.join(data_dir, dataset_name, f"{dataset_name}_stats.pt") 
        #graph
        graph_path = os.path.join(data_dir, dataset_name, f"dgl_graph_100M_int32_used")
        graph =  dgl.load_graphs(graph_path)[0][0]
        if "year" in graph.ndata:
            del graph.ndata["year"]
        if drop_model != "directed_to_undirected": 
            graph = dgl.to_bidirected(graph)        
        graph = graph.remove_self_loop().add_self_loop()

        #feat
        if "float16" in feat_type:
            feats = torch.tensor(np.load(feats_path),dtype=torch.float16) 
        else:
            feats = torch.tensor(np.load(feats_path),dtype=torch.float32) 
        if feat_type=="e5_float16":
            _mean, _ = torch.load(scale_path)
            _mean = _mean.to(torch.float16)
        if no_scale == False:
            if feat_type=="e5_float16":
                feats = feats - _mean
            else: 
                feats = center_feats(feats)
    
        feats = papers100M_feat_type(feats, used_node_path)
        #split
        split_idx={}
        split_len={}
        for name in ["train","valid","test"]:  
            split_path = os.path.join(data_dir, dataset_name, f"{short_name[dataset_name]}_{name}_split.csv.gz")
            split_idx[name] = torch.from_numpy(pd.read_csv(split_path, compression='gzip',header=None).values.reshape(-1))
            split_len[name] = split_idx[name].shape[0]
        #label
        labels = np.load(lable_path)
        labels = torch.from_numpy(labels["node_label"].reshape(-1)).int()
        train_lbls = labels[split_idx["train"]]
        val_lbls = labels[split_idx["valid"]]
        test_lbls = labels[split_idx["test"]]
        labels = torch.cat([train_lbls, val_lbls, test_lbls])
        #ego_graph
        if os.path.exists(os.path.join(data_dir, dataset_name, f"{dataset_name}-lc-ego-graphs-256-int32.npy")):
            ego_graph_path = os.path.join(data_dir, dataset_name, f"{dataset_name}-lc-ego-graphs-256-int32.npy")
            ego_graphs = load_ego_graphs(ego_graph_path, max_samples=None)
            ego_graphs = [ego_graphs[0:split_len["train"]],ego_graphs[split_len["train"]:split_len["train"]+split_len["valid"]],ego_graphs[split_len["train"]+split_len["valid"]:split_len["train"]+split_len["valid"]+split_len["test"]]]
        else:
            ego_graph_path = os.path.join(data_dir, dataset_name, f"{dataset_name}-lc-ego-graphs-256.pt") 
            ego_graphs = torch.load(ego_graphs_file_path) 
        
        return feats, graph, labels, split_idx, ego_graphs
        
    elif dataset_name in ["FB15K237", "WN18RR","Wiki","ConceptNet"]:
        #prefix = "_clean_1M"  #"_clean"
        if dataset_name =="FB15K237":
            prefix1 = "_trainedge" #FB15
            prefix2 = "" #FB15
        elif dataset_name =="ConceptNet":
            prefix1 = "" 
            prefix2 = "" 
        elif dataset_name =="Wiki":
            prefix1 = "_clean_1M" 
            prefix2 = "_clean_1M" 

        elif dataset_name =="WN18RR":
            prefix1 = "" 
            prefix2 = "" 

        feats_path = os.path.join(data_dir, dataset_name, f"{dataset_name}_embedding_{feat_type}{prefix2}.npy")
        lable_path = os.path.join(data_dir, dataset_name, f"{dataset_name}_labels_trvate{prefix2}.pt")
        ego_graph_path = os.path.join(data_dir, dataset_name, f"{dataset_name}-lc-ego-graphs-256-int32{prefix1}.pt")
        graph_path = os.path.join(data_dir, dataset_name, f"dgl_graph_{dataset_name}_trainedge_int32{prefix2}")
        #graph_path = os.path.join(data_dir, dataset_name, f"dgl_graph_{dataset_name}_fulledge_int32")
        split_path = os.path.join(data_dir, dataset_name, f"{dataset_name}_data_split{prefix2}.json")

        #assert feat_type=="e5_float16"
        if os.path.exists(feats_path) == False:
            raise  FileNotFoundError(f"{feats_path} doesn't exist")
        feats = torch.tensor(np.load(feats_path),dtype=torch.float16) 
        if no_scale == False:    
            feats = center_feats(feats)
        
        if os.path.exists(graph_path) == False:
            raise  FileNotFoundError(f"{graph_path} doesn't exist")
        graph = dgl.load_graphs(graph_path)[0][0]    
        #graph = dgl.to_bidirected(graph)
        graph = graph.remove_self_loop().add_self_loop() 
        if "_ID" in graph.ndata:
            del graph.ndata["_ID"] 
        if "_ID" in graph.edata:
            del graph.edata["_ID"]
    
        if os.path.exists(lable_path):
            labels = torch.load(lable_path)
        else:
            labels = None
        
        if os.path.exists(split_path):
            with open(split_path, 'r') as json_file:
                split_idx_tmp = json.load(json_file)
            split_idx = {}
            for k,v in split_idx_tmp.items():
                split_idx[k] = torch.tensor(v, dtype=torch.int64)
        else:
            split_idx = None

        if os.path.exists(ego_graph_path):
            ego_graphs = torch.load(ego_graph_path)
        else:
            ego_graphs = None

        return feats, graph, labels, split_idx, ego_graphs
    
    elif dataset_name in ["Pubmed", "Cora"]:
        feats_path = os.path.join(data_dir, dataset_name, f"{dataset_name}_embedding_{feat_type}.npy")
        lable_path = os.path.join(data_dir, dataset_name, f"{dataset_name}_labels.pt")
        ego_graph_path = None
        graph_path = os.path.join(data_dir, dataset_name, f"dgl_graph_{dataset_name}_undirect_int32")
        split_path = os.path.join(data_dir, dataset_name, f"{dataset_name}_data_split.json")

        #assert feat_type=="e5_float16"
        feats = torch.tensor(np.load(feats_path),dtype=torch.float16) 
        if no_scale == False:    
            feats = center_feats(feats)

        graph = dgl.load_graphs(graph_path)[0][0]    
        graph = dgl.to_bidirected(graph)
        graph = graph.remove_self_loop().add_self_loop() 
    
        if os.path.exists(split_path):
            with open(split_path, 'r') as json_file:
                split_idx_tmp = json.load(json_file)
            split_idx = {}
            for k,v in split_idx_tmp.items():
                split_idx[k] = torch.tensor(v, dtype=torch.int64)
        else:
            split_idx = None

        labels = torch.load(lable_path)
        train_lbls = labels[split_idx["train"]]
        val_lbls = labels[split_idx["valid"]]
        test_lbls = labels[split_idx["test"]]
        labels = torch.cat([train_lbls, val_lbls, test_lbls])
        ego_graphs = None
        return feats, graph, labels, split_idx, ego_graphs

    else:
        raise NotImplementedError




def load_ego_graphs(ego_graph_path, max_samples=None):
    ego_graphs = np.load(ego_graph_path)
    ego_graphs = convert_1d_array_to_ego_graph_list(ego_graphs)
    print(f"num_ego_graphs: {len(ego_graphs)}")
    return ego_graphs



def center_feats(x):
    scaler = StandardScaler(with_std=False)
    #print("only centering")    
    origin_type = x.dtype
    feats = x.numpy()
    scaler.fit(feats)
    feats = torch.from_numpy(scaler.transform(feats)).to(origin_type)
    return feats

def norm_feats(x):
    scaler = StandardScaler()
    origin_type = x.dtype
    feats = x.numpy()
    scaler.fit(feats)
    feats = torch.from_numpy(scaler.transform(feats)).to(origin_type)
    return feats