import wget
import os
import argparse
import gzip
import shutil
from ogb.nodeproppred import DglNodePropPredDataset
import zipfile
import torch
from torch.utils.data import Dataset , DataLoader
from transformers import AutoTokenizer , AutoModel
import pandas as pd
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import gdown
import dgl
from scipy.sparse import csr_matrix
from localgraphclustering import *
from collections import namedtuple
import multiprocessing
import os.path as osp
import json
from sklearn.preprocessing import StandardScaler

MODEL_NAME={"e5":"intfloat/e5-small-v2", "ofa": "../../cache/transformer-model/multi-qa-distilbert-cos-v1"}
short_name = {"ogbn-papers100M":"100M","ogbn-arxiv":"arxiv","ogbn-products":"products"}
def decompress_gz(file_path, output_path):
    with gzip.open(file_path, 'rb') as f_in:
        with open(output_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    print(f"Decompressed file to {output_path}")

def unzip_file(zip_file_path, extract_to_path):
    try:
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to_path)
        print(f"Extracted {zip_file_path} to {extract_to_path} successfully")
    except zipfile.BadZipFile:
        print(f"Error: {zip_file_path} is not a valid ZIP file")
    except Exception as e:
        print(f"Error extracting {zip_file_path}: {e}")

def move_folder_contents(src_folder, dest_folder):
    if not os.path.exists(src_folder):
        print(f"{src_folder} not exit")
        return
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    for item in os.listdir(src_folder):
        src_path = os.path.join(src_folder, item)
        dest_path = os.path.join(dest_folder, item)
        shutil.move(src_path, dest_path)

def convert_1d_array_to_ego_graph_list(path_or_data, salt=-200000000):
    if type(path_or_data) == str:
        array = np.load(path_or_data)
    else:
        array = path_or_data
    index = np.where(array < 0)[0]
    array[index] -= salt
    ego_graphs = np.split(array, index[1:])
    return ego_graphs

class Ogb_dataset(Dataset):
    def __init__(self, datas): 
        self.data = datas
        self.length = len(self.data)
       
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self): 
        return self.length 
    
class Tokenizer(object):
    def __init__(self, tokenizer,args):
        super(Tokenizer, self).__init__()   
        self.max_token_len = args.max_token_len
        self.tokenizer = tokenizer
        self.padding = "max_length"
        self.truncation = True
    
    def __call__(self, examples):
        if isinstance(examples, str):
            return self.tokenizer(examples, padding=self.padding, truncation=self.truncation ,max_length=self.max_token_len, return_tensors="pt")
        else:
            return self.tokenizer(examples["text"], padding=self.padding, truncation=self.truncation ,max_length=self.max_token_len, return_tensors="pt")

def my_sweep_cut(g, node):
    vol_sum = 0.0
    in_edge = 0.0
    conds = np.zeros_like(node, dtype=np.float32)
    for i in range(len(node)):
        idx = node[i]
        vol_sum += g.d[idx]
        denominator = min(vol_sum, g.vol_G - vol_sum)
        if denominator == 0.0:
            denominator = 1.0
        in_edge += 2*sum([g.adjacency_matrix[idx,prev] for prev in node[:i+1]])
        cut = vol_sum - in_edge
        conds[i] = cut/denominator
    return conds

def calc_local_clustering(args):
    i, log_steps, num_iter, ego_size, method = args
    if i % log_steps == 0:
        print(i)
    node, ppr = approximate_PageRank(graphlocal, [i], iterations=num_iter, method=method, normalize=False)
    d_inv = graphlocal.dn[node]
    d_inv[d_inv > 1.0] = 1.0
    ppr_d_inv = ppr * d_inv
    output = list(zip(node, ppr_d_inv))[:ego_size]
    node, ppr_d_inv = zip(*sorted(output, key=lambda x: x[1], reverse=True))
    assert node[0] == i
    node = np.array(node, dtype=np.int32)
    conds = my_sweep_cut(graphlocal, node)
    return node, conds

def norm_feats(x):
    scaler = StandardScaler()
    feats = x
    scaler.fit(feats)
    mean = scaler.mean_
    std = scaler.scale_
    return  mean, std


class Gen_ogb_data():
    def __init__(self,args):
        self.args = args
        self.download_raw_data(args.dataset_name)
        self.get_text_data(args.dataset_name)
        self.get_emb(args.dataset_name)
        self.get_graph_label_split_scale(args.dataset_name)
        self.localclustering(args.dataset_name)
        if args.dataset_name == "ogbn-papers100M":
            self.reduce_100M_memory_cost(args.dataset_name)

    def average_pool(self, last_hidden_states,
                    attention_mask):
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def download_raw_data(self,dataset_name="ogbn-arxiv"):
        os.makedirs(self.args.data_save_path, exist_ok=True)
        dataset_path = os.path.join(self.args.data_save_path, dataset_name)
        os.makedirs(dataset_path, exist_ok=True)
        if dataset_name == "ogbn-arxiv":
            # downlaod ogbn-data raw text
            if not os.path.exists(os.path.join(dataset_path, "titleabs.tsv")):
                url = "https://snap.stanford.edu/ogb/data/misc/ogbn_arxiv/titleabs.tsv.gz"
                wget.download(url, os.path.join(dataset_path, "titleabs.tsv.gz"))
                decompress_gz(os.path.join(dataset_path, "titleabs.tsv.gz"),os.path.join(dataset_path, "titleabs.tsv"))
                os.remove(os.path.join(dataset_path, "titleabs.tsv.gz"))
            self.dgl_dataset = DglNodePropPredDataset(dataset_name, root=os.path.join(self.args.data_save_path, "ogb-official-data"))

        elif dataset_name == "ogbn-products":
            # downlaod ogbn-data raw text
            if not os.path.exists(os.path.join(dataset_path, "tst.json")):
                gdown.download("https://drive.google.com/uc?id=1gsabsx8KR2N9jJz16jTcA0QASXsNuKnN", os.path.join(dataset_path, "Amazon-3M.raw.zip"), quiet=False)
                unzip_file(os.path.join(dataset_path, "Amazon-3M.raw.zip"), os.path.join(dataset_path, "Amazon-3M"))
                decompress_gz(os.path.join(dataset_path, "Amazon-3M", "Amazon-3M.raw", "trn.json.gz"), os.path.join(dataset_path, "trn.json"))
                decompress_gz(os.path.join(dataset_path, "Amazon-3M", "Amazon-3M.raw", "tst.json.gz"), os.path.join(dataset_path, "tst.json"))
                os.remove(os.path.join(dataset_path, "Amazon-3M.raw.zip"))
                shutil.rmtree(os.path.join(dataset_path, "Amazon-3M"))
            self.dgl_dataset = DglNodePropPredDataset(dataset_name, root=os.path.join(self.args.data_save_path, "ogb-official-data"))

        elif dataset_name == "ogbn-papers100M":
            if not os.path.exists(os.path.join(dataset_path, "paperinfo")):
                wget.download("https://snap.stanford.edu/ogb/data/misc/ogbn_papers100M/paperinfo.zip", os.path.join(dataset_path, "paperinfo.zip"))
                unzip_file(os.path.join(dataset_path, "paperinfo.zip"), os.path.join(dataset_path, "paperinfo"))
                os.remove(os.path.join(dataset_path, "paperinfo.zip"))
            self.dgl_dataset = DglNodePropPredDataset(dataset_name, root=os.path.join(self.args.data_save_path, "ogb-official-data"))

    def get_text_data(self, dataset_name="ogbn-arxiv"):
        dataset_path = os.path.join(self.args.data_save_path, dataset_name)
        ogbn_official_path = os.path.join(self.args.data_save_path, "ogb-official-data")

        if dataset_name=="ogbn-products":
            decompress_gz(os.path.join(ogbn_official_path,"ogbn_products","mapping",'nodeidx2asin.csv.gz'),os.path.join(ogbn_official_path,"ogbn_products","mapping",'nodeidx2asin.csv'))
            self.nodeid2contentid = pd.read_csv(os.path.join(ogbn_official_path,"ogbn_products","mapping",'nodeidx2asin.csv')) #(2449029*2)
            self.df = {"contentid":[],"title":[],"content":[]}
            for line in open(os.path.join(dataset_path,"trn.json")):
                one_dict=json.loads(line)
                self.df["contentid"].append(one_dict["uid"])
                self.df["title"].append(one_dict["title"])
                self.df["content"].append(one_dict["content"])
            
            for line in open(os.path.join(dataset_path,"tst.json")) :
                one_dict=json.loads(line)
                self.df["contentid"].append(one_dict["uid"])
                self.df["title"].append(one_dict["title"])
                self.df["content"].append(one_dict["content"])
            self.df = pd.DataFrame(self.df)
            self.df.columns = ["paperid", "title", "abs"]
            self.nodeid2contentid.columns = ["nodeid", "paperid"]
            data = pd.merge(self.nodeid2contentid, self.df, how="left", on="paperid")  
            Datasets = data.values[:,2:]

        elif dataset_name=="ogbn-arxiv":
            self.df = pd.read_csv(os.path.join(dataset_path,'titleabs.tsv'), sep='\t')
            decompress_gz(os.path.join(ogbn_official_path,"ogbn_arxiv","mapping",'nodeidx2paperid.csv.gz'),os.path.join(ogbn_official_path,"ogbn_arxiv","mapping",'nodeidx2paperid.csv'))
            self.nodeid2contentid = pd.read_csv(os.path.join(ogbn_official_path,"ogbn_arxiv","mapping",'nodeidx2paperid.csv'))
            self.df.columns = ["paperid", "title", "abs"]
            self.nodeid2contentid.columns = ["nodeid", "paperid"]
            data = pd.merge(self.nodeid2contentid, self.df, how="left", on="paperid")  
            Datasets = data.values[:,2:]

        elif dataset_name=="ogbn-papers100M":
            abstract = pd.read_csv(os.path.join(dataset_path, "paperinfo","idx_abs.tsv"), sep='\t', header=None)
            title = pd.read_csv(os.path.join(dataset_path, "paperinfo", "idx_title.tsv"), sep='\t', header=None)
           
            title.columns = ["ID", "Title"]
            title["ID"] = title["ID"].astype(np.int64)
            abstract.columns = ["ID", "Abstract"]
            abstract["ID"] = abstract["ID"].astype(np.int64)
            data = pd.merge(title, abstract, how="outer", on="ID", sort=True)
            
            paper_id_path_csv = os.path.join(ogbn_official_path,"ogbn_papers100M","mapping", "nodeidx2paperid.csv.gz")  
            paper_ids = pd.read_csv(paper_id_path_csv, usecols=[0])
            paper_ids.columns = ["ID"]

            data.columns = ["ID", "Title", "Abstract"]
            data["ID"] = data["ID"].astype(np.int64)
            data = pd.merge(paper_ids, data, how="left", on="ID")  
            Datasets = data.values[:,1:]
        
        dataframe = pd.DataFrame(Datasets)
        dataframe.to_csv(os.path.join(dataset_path,f'{dataset_name}_title_content.csv'),index=False)
        print(f"{dataset_name} title_content.csv has been saved!")
            
    def get_emb(self, dataset_name="ogbn-arxiv"):
        dataset_path = os.path.join(self.args.data_save_path, dataset_name)
        if dataset_name in ["ogbn-arxiv","ogbn-products","ogbn-papers100M"]:
            Datas = []
            data = pd.read_csv(os.path.join(dataset_path,f'{dataset_name}_title_content.csv')).values
            for k in range(data.shape[0]):
                data_dict = {}
                if pd.isnull(data[k][0]) and pd.isnull(data[k][1]):
                    data_dict["text"] = " .  "
                elif pd.isnull(data[k][1]):
                    data_dict["text"] = data[k][0]
                elif pd.isnull(data[k][0]):
                    data_dict["text"] = data[k][1]
                else:
                    data_dict["text"] = data[k][0]+". "+data[k][1]
                Datas.append(data_dict)
            text_dataset = Ogb_dataset(Datas)
            text_dataloader = DataLoader(text_dataset, shuffle=False, batch_size=self.args.batch_size)
        else:
            raise ValueError
        

        if self.args.Model == "e5":
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME[self.args.Model])
            model_tokenizer = Tokenizer(tokenizer, self.args)
            model = AutoModel.from_pretrained(MODEL_NAME[self.args.Model])
            model.to(self.args.device)
            model.eval()
            nodes_embed=[]
            epoch_iter = tqdm(text_dataloader)
            print(f"Generating {self.args.Model} embedding!")
            with torch.no_grad():
                for batch in epoch_iter:
                    batch = model_tokenizer(batch)
                    batch = {k: v.to(self.args.device) for k, v in batch.items()}
                    outputs = model(**batch)
                    embeddings = self.average_pool(outputs.last_hidden_state, batch['attention_mask'])
                    for i in range(embeddings.shape[0]):
                        nodes_embed.append(embeddings[i].cpu().numpy().astype(self.args.dtype))

            nodes_embed = np.stack(nodes_embed,axis=0)
            np.save(os.path.join(dataset_path,f"{short_name[dataset_name]}_embedding_{self.args.Model}_{self.args.dtype}.npy"), nodes_embed)
            print(f"{short_name[dataset_name]}_embedding_{self.args.Model}_{self.args.dtype}.npy has been saved!")
        
        elif self.args.Model == "ofa":
            model = SentenceTransformer(MODEL_NAME[self.args.Model])
            model.to(self.args.device)
            model.eval()
            with torch.no_grad():
                texts = []
                for d in Datas:
                    texts.append(d["text"])
                embeddings = model.encode(texts, batch_size=self.args.batch_size, show_progress_bar=True, convert_to_tensor=False, convert_to_numpy=not to_tensor )
                np.save(os.path.join(dataset_path,f"{short_name[dataset_name]}_embedding_{self.args.Model}_{self.args.dtype}.npy"), embeddings.astype(self.args.dtype))
                print(f"{short_name[dataset_name]}_embedding_{self.args.Model}_{self.args.dtype}.npy has been saved!")
        else:
            raise ValueError
    
    def localclustering(self, dataset_name="ogbn-arxiv"):
        dataset_path = os.path.join(self.args.data_save_path, dataset_name)
        np.random.seed(0)

        graph, label = self.dgl_dataset[0]
        if "year" in graph.ndata:
            del graph.ndata["year"]
        if not graph.is_multigraph:
            graph = dgl.to_bidirected(graph)   
            graph = graph.remove_self_loop().add_self_loop()
        split_idx = self.dgl_dataset.get_idx_split()

        save_path = os.path.join(dataset_path, f"{dataset_name}-lc-ego-graphs-256.pt") 
       
        N = graph.num_nodes()  
        edge_index = graph.edges()
        edge_index = (edge_index[0].numpy(), edge_index[1].numpy())
        adj = csr_matrix((np.ones(edge_index[0].shape[0]), edge_index), shape=(N, N))

        global graphlocal
        graphlocal = GraphLocal.from_sparse_adjacency(adj)
        print('graphlocal generated')

        train_idx = split_idx["train"].cpu().numpy()
        valid_idx = split_idx["valid"].cpu().numpy()
        test_idx = split_idx["test"].cpu().numpy()

        ego_size=256
        num_iter=1000
        log_steps=10000
        num_workers=32
        method='acl'

        with multiprocessing.Pool(num_workers) as pool:
            ego_graphs_train, conds_train = zip(*pool.imap(calc_local_clustering, [(i, log_steps, num_iter, ego_size, method) for i in train_idx], chunksize=512))

        with multiprocessing.Pool(num_workers) as pool:
            ego_graphs_valid, conds_valid = zip(*pool.imap(calc_local_clustering, [(i, log_steps, num_iter, ego_size, method) for i in valid_idx], chunksize=512))

        with multiprocessing.Pool(num_workers) as pool:
            ego_graphs_test, conds_test = zip(*pool.imap(calc_local_clustering, [(i, log_steps, num_iter, ego_size, method) for i in test_idx], chunksize=512))

        ego_graphs = [ego_graphs_train, ego_graphs_valid, ego_graphs_test]
        torch.save(ego_graphs, save_path)
        print(f"{dataset_name}-lc-ego-graphs-256.pt has been saved!")

    def get_graph_label_split_scale(self, dataset_name="ogbn-arxiv"):
        dataset_path = os.path.join(self.args.data_save_path, dataset_name)
        #graph
        graph, label = self.dgl_dataset[0]
        graph = dgl.graph((graph.edges()[0].to(torch.int32), graph.edges()[1].to(torch.int32)), num_nodes=graph.number_of_nodes())
        dgl.save_graphs(os.path.join(dataset_path,f"dgl_graph_{short_name[dataset_name]}_int32"),graph)
        #label
        if dataset_name in ["ogbn-arxiv","ogbn-products"]:
            torch.save(label.reshape(-1), os.path.join(dataset_path, f'{short_name[dataset_name]}_label.pt'))
        elif dataset_name in ["ogbn-papers100M"]:
            shutil.copy(os.path.join(ogbn_official_path,"ogbn_papers100M","raw", "node-label.npz")  , os.path.join(dataset_path, "100M-node-label.npz"))
        #split
        if dataset_name in ["ogbn-arxiv","ogbn-products"]:
            split_idx = self.dgl_dataset.get_idx_split()
            torch.save(split_idx, os.path.join(dataset_path, f'{short_name[dataset_name]}_split.pt'))
        elif dataset_name in ["ogbn-papers100M"]:
            for n in ["train","test","valid"]:
                shutil.copy(os.path.join(ogbn_official_path,"ogbn_papers100M","split", "time", f"{n}.csv.gz")  , os.path.join(dataset_path, f"100M_{n}_split.csv.gz"))
        #scale 
        mean, std = norm_feats(np.load(os.path.join(dataset_path,f"{short_name[dataset_name]}_embedding_{self.args.Model}_{self.args.dtype}.npy")))
        torch.save((torch.from_numpy(mean), torch.from_numpy(std)), os.path.join(dataset_path, f'{dataset_name}_stats.pt')) 
      

    def reduce_100M_memory_cost(self,dataset_name):
        dataset_path = os.path.join(self.args.data_save_path, dataset_name)
        #cal repeat node:
        if os.path.exists(os.path.join(dataset_path,"ogbn-papers100M-lc-ego-graphs-256-int32.npy")):
            ego_graph = np.load(os.path.join(dataset_path,"ogbn-papers100M-lc-ego-graphs-256-int32.npy"))
            ego_graph = convert_1d_array_to_ego_graph_list(ego_graph)
        else:
            assert os.path.exists(os.path.join(dataset_path,"ogbn-papers100M-lc-ego-graphs-256.pt"))
            ego_graph = torch.load(os.path.join(dataset_path,"ogbn-papers100M-lc-ego-graphs-256.pt"))
            ego_graph = ego_graph[0]+ego_graph[1]+ego_graph[2]
        unique_numbers = set()
        for g in ego_graph:
            unique_numbers.update(g)
        unique_numbers_list = sorted(list(unique_numbers))
        unique_numbers_list = np.array(unique_numbers_list)
        np.save(os.path.join(dataset_path,"used_node.npy"), unique_numbers_list)
        #reduce feature
        feat = np.load(os.path.join(dataset_path,f"{short_name[dataset_name]}_embedding_{self.args.Model}_{self.args.dtype}.npy"))
        feat = feat[unique_numbers_list]
        np.save(os.path.join(dataset_path,f"{short_name[dataset_name]}_embedding_{self.args.Model}_{self.args.dtype}_used.npy"), feat)
        #reduce graph
        graph = dgl.load_graphs(os.path.join(dataset_path,f"dgl_graph_{short_name[dataset_name]}_int32"))[0][0]
        src, dst = graph.edges()
        src = src.numpy()
        dst = dst.numpy()
        mask1 = np.isin(src, unique_numbers_list)
        mask2 = np.isin(dst, unique_numbers_list)
        tot_mask = mask1*mask2
        new_src = src[tot_mask]
        new_dst = dst[tot_mask]
        new_g = dgl.DGLGraph()
        new_g.add_nodes(graph.number_of_nodes())
        new_g.add_edges(new_src,new_dst)
        dgl.save_graphs(os.path.join(dataset_path,f"dgl_graph_{short_name[dataset_name]}_int32_used"),new_g)
        

class Gen_fewshot_data():
    def __init__(self,args):
        self.args = args
        self.dataset_path = os.path.join(args.data_save_path, args.dataset_name)
        self.move_raw_data()
        if self.args.dataset_name in ["FB15K237", "WN18RR"]:
            self.bulid_entityid_to_description()
            self.bulid_graph_feats_label_split_data()
        elif self.args.dataset_name in ["Cora"]:
            self.cora_bulid_graph_feats_label_split_data()
    
    def move_raw_data(self):
        os.makedirs(self.args.data_save_path, exist_ok=True)
        dataset_path = os.path.join(self.args.data_save_path, self.args.dataset_name)
        os.makedirs(dataset_path, exist_ok=True)
        move_folder_contents(f"./fewshot_data/{self.args.dataset_name}",dataset_path)
    
    def average_pool(self, last_hidden_states,attention_mask):
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def bulid_entityid_to_description(self):
        self.entity_lst = []
        self.text_lst = []
        
        if self.args.dataset_name == "FB15K237":
            with open(osp.join(self.dataset_path, "entity2wikidata.json"), "r") as f:
                data = json.load(f)
            for k in data:
                self.entity_lst.append(k)
                self.text_lst.append(
                    "entity nammes: "
                    + data[k]["label"]
                    + ", entity alternatives: "
                    + ", ".join(data[k]["alternatives"])
                    + ". entity descriptions:"
                    + data[k]["description"]
                    if data[k]["description"] is not None
                    else "None"
                )
            self.entity2id = {entity: i for i, entity in enumerate(self.entity_lst)}

        elif self.args.dataset_name == "WN18RR":
            with open(osp.join(self.dataset_path, "entity2text.txt"), "r") as f:
                lines = f.readlines()
                for line in lines:
                    tmp = line.strip().split("\t")
                    self.entity_lst.append(tmp[0])
                    self.text_lst.append(tmp[1])
            self.entity2id = {entity: i for i, entity in enumerate(self.entity_lst)}

    def bulid_graph_feats_label_split_data(self):
        names = ["train", "valid", "test"]
        name_dict = {n: osp.join(self.dataset_path, n + ".txt") for n in names}
        
        relation2id = {}
        converted_triplets = {}
        rel_list = []
        rel = len(relation2id)

        for file_type, file_path in name_dict.items():
            edges = []
            edge_types = []
            with open(file_path) as f:
                file_data = [line.split() for line in f.read().split("\n")[:-1]]
            unknown_entity = 0
            for triplet in file_data:
                if triplet[0] not in self.entity2id:
                    self.text_lst.append("entity names: Unknown")
                    self.entity_lst.append(triplet[0])
                    self.entity2id[triplet[0]] = len(self.entity2id)
                    unknown_entity += 1
                if triplet[2] not in self.entity2id:
                    self.text_lst.append("entity names: Unknown")
                    self.entity_lst.append(triplet[2])
                    self.entity2id[triplet[2]] = len(self.entity2id)
                    unknown_entity += 1
                if triplet[1] not in relation2id:
                    relation2id[triplet[1]] = rel
                    rel_list.append(triplet[1])
                    rel += 1
                edges.append(
                    [
                        self.entity2id[triplet[0]],
                        self.entity2id[triplet[2]],
                    ]
                )
                edge_types.append(relation2id[triplet[1]])
            print(file_type+" unknown_entity:", unknown_entity)
            converted_triplets[file_type] = [edges, edge_types]
  
        #graph feats label split 

        #bulid graph
        num_nodes = len(self.entity2id)
        self.total_nodes = num_nodes
        for graph_type in ["train_edge","full_edge"]:
            if graph_type=="train_edge":
                edges = torch.tensor(converted_triplets["train"][0]).T
                graph = dgl.graph((edges[0].to(torch.int32), edges[1].to(torch.int32)), num_nodes=num_nodes)
                print(graph)
                dgl.save_graphs(os.path.join(self.dataset_path, f"dgl_graph_{self.args.dataset_name}_trainedge_int32"), graph)
            elif graph_type=="full_edge":
                edges = torch.tensor(converted_triplets["train"][0]+converted_triplets["valid"][0]+converted_triplets["test"][0]).T
                graph = dgl.graph((edges[0].to(torch.int32), edges[1].to(torch.int32)), num_nodes=num_nodes)
                print(graph)
                dgl.save_graphs(os.path.join(self.dataset_path, f"dgl_graph_{self.args.dataset_name}_fulledge_int32") ,graph)

        #bulid feats
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME[self.args.Model])
        model_tokenizer = Tokenizer(tokenizer, self.args)
        model = AutoModel.from_pretrained(MODEL_NAME[self.args.Model])       
        model.to(self.args.device)
        model.eval()
        nodes_embed=[]
        with torch.no_grad():
            for data in self.text_lst:
                batch = model_tokenizer(data)
                batch = {k: v.to(self.args.device) for k, v in batch.items()}
                outputs = model(**batch)
                embeddings = self.average_pool(outputs.last_hidden_state, batch['attention_mask'])
                for i in range(embeddings.shape[0]):
                    nodes_embed.append(embeddings[i].cpu().numpy().astype(self.args.dtype)) 

            nodes_embed=np.stack(nodes_embed,axis=0)
            data_len = nodes_embed.shape[0]
            assert self.total_nodes == data_len
        np.save(os.path.join(self.dataset_path, f"{self.args.dataset_name}_embedding_{self.args.Model}_{self.args.dtype}.npy"), nodes_embed)
       
        #bulid labels
        labels = torch.tensor(converted_triplets["train"][1]+converted_triplets["valid"][1]+converted_triplets["test"][1] ,dtype=torch.int64) 
        print("labels shape:",labels.shape)
        torch.save(labels, os.path.join(self.dataset_path, f"{self.args.dataset_name}_labels_trvate.pt"))

        #bulid split:
        splits = {"train":converted_triplets["train"][0],"valid":converted_triplets["valid"][0],"test":converted_triplets["test"][0]}
        with open(os.path.join(self.dataset_path,f"{self.args.dataset_name}_data_split.json"), 'w') as json_file:
            json.dump(splits, json_file)

    def cora_bulid_graph_feats_label_split_data(self):
        data = torch.load(os.path.join(self.dataset_path, "cora.pt"))
        #graph
        edges = data.edge_index
        g = dgl.graph((edges[0].to(torch.int32),edges[1].to(torch.int32)), num_nodes=len(data.raw_texts))
        self.total_nodes = g.num_nodes()
        g = dgl.to_bidirected(g)
        dgl.save_graphs(os.path.join(self.dataset_path, f"dgl_graph_{self.args.dataset_name}_undirect_int32") ,g)
        
        #bulid feats
        text = data.raw_texts
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME[self.args.Model])
        model_tokenizer = Tokenizer(tokenizer, self.args)
        model = AutoModel.from_pretrained(MODEL_NAME[self.args.Model])       
        model.to(self.args.device)
        model.eval()
        nodes_embed=[]
        with torch.no_grad():
            for d in text:
                batch = model_tokenizer(d)
                batch = {k: v.to(self.args.device) for k, v in batch.items()}
                outputs = model(**batch)
                embeddings = self.average_pool(outputs.last_hidden_state, batch['attention_mask'])
                for i in range(embeddings.shape[0]):
                    nodes_embed.append(embeddings[i].cpu().numpy().astype(self.args.dtype)) 

            nodes_embed=np.stack(nodes_embed,axis=0)
            data_len = nodes_embed.shape[0]
            assert self.total_nodes == data_len
        np.save(os.path.join(self.dataset_path, f"{self.args.dataset_name}_embedding_{self.args.Model}_{self.args.dtype}.npy"), nodes_embed)

        #bulid labels
        labels = data.y
        print("labels shape:",labels.shape)
        torch.save(labels, os.path.join(self.dataset_path, "Cora_labels.pt"))
        
        #bulid split:
        splits = {"train":np.where(data.train_masks[0].numpy()==True)[0].tolist() ,"valid": np.where(data.val_masks[0].numpy()==True)[0].tolist() ,"test":np.where(data.test_masks[0].numpy()==True)[0].tolist()}
        with open(os.path.join(self.dataset_path,"Cora_data_split.json"), 'w') as json_file:
            json.dump(splits, json_file)
        

parser = argparse.ArgumentParser(description="")
parser.add_argument('--data_save_path', type=str, default='./data')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--Model', type=str, default='e5')
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--dataset_name', type=str, nargs="+", default=["ogbn-arxiv","ogbn-products","ogbn-papers100M","FB15K237","Cora","WN18RR"])
parser.add_argument('--max_token_len', type=int, default=512)
parser.add_argument('--dtype', type=str, default="float16")

if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    total_datasets_to_process = args.dataset_name
    for dn in total_datasets_to_process:
        if dn in ["ogbn-arxiv","ogbn-products","ogbn-papers100M"]:
            args.dataset_name = dn
            Gen_ogb_data(args)
        elif dn in ["FB15K237","Cora","WN18RR"]:
            args.dataset_name = dn
            Gen_fewshot_data(args)

