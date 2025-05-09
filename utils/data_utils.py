import os
import pickle
import queue
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle
import gzip
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from dgl.dataloading import GraphDataLoader
from ogb.nodeproppred import DglNodePropPredDataset
from sentence_transformers import SentenceTransformer
from torch import Tensor
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from src.trainer_large import ModelTrainer, build_model
from src.utils.utils import set_random_seed

# Global variables
dataset_names = ['ogbn-mag','ogbn-arxiv','ogbn-products','combined']
utils_path = os.path.dirname(os.path.abspath(__file__))
cwd = os.path.dirname(utils_path)
model_dir = os.path.join(cwd, 'model')
data_dir = os.path.join(cwd, 'data')
model_path = os.path.join(model_dir, 'GraphAlign_graphmae.pt')
stark_path = os.path.join(data_dir, 'stark-mag')

# mapping values
STARK_FILES = {
    "edge_index": "edge_index.pt",
    "edge_types": "edge_types.pt",
    "node_type_dict": "node_type_dict.pkl",
    "edge_type_dict": "edge_type_dict.pkl",
    "node_info": "node_info.pkl",
    "node_types": "node_types.pt"
}
MAPPING_FILES = {
    'ogbn-arxiv': os.path.join(data_dir, 'nodeidx2paperid.csv.gz'),
    'ogbn-mag': os.path.join(data_dir, 'paper_entidx2name.csv.gz')
}

# Model parameters
MODEL_NAME={"e5":"intfloat/e5-small-v2"}
MODEL_DIMS={"e5":384}

# get torch device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_ogb_dataset(name):
    """
    Load  OGB dataset.
    Input: dataset name
    Output: graph, labels, train_idx, valid_idx, test_idx
    """
    data_dir = os.path.join(cwd, 'data')
    print(f"Dataset directory: {data_dir}")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if name not in dataset_names:
        raise ValueError(f"Dataset {name} not supported. Supported datasets are: {dataset_names}")
    else:
        dataset = DglNodePropPredDataset(name, root=data_dir)
        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        return dataset, train_idx, valid_idx, test_idx

'''
Adapted from GraphAlign Github repository
'''

class GraphAlign_e5(ModelTrainer):
    def __init__(self, args):
        super().__init__(args)
        self._args = args
        self.pretrain_seed = 42
        if self._args.dataset == 'combined':
            graph_path = os.path.join(data_dir,'combined_graph_paper_id.pkl.gz')
            with gzip.open(graph_path, 'rb') as f:
                graph = pickle.load(f)
                self.graph = graph.to(args.device)
                self.label = graph.ndata['label']
                self.train_idx = np.where(graph.ndata['train_mask'] == 1)[0]
                self.val_idx = np.where(graph.ndata['valid_mask'] == 1)[0]
                self.test_idx = np.where(graph.ndata['test_mask'] == 1)[0]
                self.feat_type = getattr(args, 'feat_type', None)
        else:
            self.dataset, self.train_idx, self.valid_idx, self.test_idx = load_ogb_dataset(args.dataset)
            self.graph, self.label = self.dataset[0]
            self.feat_type = getattr(args, 'feat_type', None)
            self.prepare_dataset()

    def load_model(self):
            set_random_seed(self.pretrain_seed)
            if self.feat_type == 'e5_float16':
                self._args.num_features = 384

            #build model
            self.model = build_model(self._args)
            self.model.to(self._args.device)

            #load model data
            print(f"Loading model from {self._args.load_model_path}")
            self.model.load_state_dict(torch.load(self._args.load_model_path))

    def get_nodeidx_mappings(self, return_val = True):
        if self._args.dataset == 'ogbn-arxiv':
            node_idx_file = MAPPING_FILES[self._args.dataset]
            node_idx = pd.read_csv(
                node_idx_file,
                sep=',', compression='gzip',
                header=0,
                names=['node_idx', 'paper_id']
            )
            node_idx['paper_id'] = node_idx['paper_id'].astype(np.int64)
            idx_title = pd.read_csv(
                os.path.join(data_dir,'titleabs.tsv.gz'),
                sep='\t',
                compression='gzip',
                header=0,
                names=['title', 'abstract'],
                dtype={'title': str, 'abstract': str}
            )
            self.node_titles = node_idx.merge(idx_title.reset_index(), left_on='paper_id', right_on='index', how='inner')
            self.paperid = torch.tensor(node_idx['paper_id'], dtype=torch.int64).to(device)
            #Here "inner" may cause problem when processing dataset mag, better use "left", and check whether how many None in the title and abstract attribute.
            if return_val:
                return self.node_titles
        elif self._args.dataset == 'ogbn-mag':
            #load file
            idx_path = os.path.join(data_dir, 'mag-nodeidx2titles.csv')
            df = pd.read_csv(
                idx_path,
                sep=',',
                header= 0
            )

            #get node_idx and OriginalTitle
            self.node_titles = df[['node_idx', 'paper_id', 'OriginalTitle']]
            self.node_titles.rename(columns={'OriginalTitle': 'title'}, inplace=True)
            self.paperid = torch.tensor(df['paper_id'],dtype=torch.int64).to(device)
            self.node_titles['node_idx'] = self.node_titles['node_idx'].astype(np.int64)

            if return_val:
                return self.node_titles
        elif self._args.dataset == 'combined':
            #load file
            idx_path = os.path.join(data_dir, 'combined-nodeidx2titles.csv')
            df = pd.read_csv(
                idx_path,
                sep=',',
                header= 0
            )
            self.node_titles = df[['paper_id', 'OriginalTitle']]
            self.node_titles.rename(columns={'OriginalTitle': 'title'}, inplace=True)
            self.paperid = torch.tensor(self.graph.ndata['paper_id'],dtype=torch.int64).to(device)

            if return_val:
                return self.node_titles

        else:
            raise ValueError(f"Dataset {self._args.dataset} not supported.")

    def generate_e5_embeddings(self, return_val = True):
        print("Generating E5 embeddings...")
        # Load the E5 model and tokenizer

        #ogbn doesnt have abstract column so I will just work with titles
        #self.node_titles['text'] = self.node_titles['title'] + ' ' + self.node_titles['abstract'].fillna('')

        self.node_titles['text'] = "query: " + self.node_titles['title']
        input_texts = self.node_titles['text'].to_list()
        model_name = MODEL_NAME["e5"]
        model = SentenceTransformer(model_name, device = device)
        batch_size = self._args.batch_size_f
        embeddings = []
        for i in tqdm(range(0, len(input_texts), batch_size), desc="Encoding Progress"):
            batch = input_texts[i:i + batch_size]
            try:
                batch_embeddings = model.encode(batch, normalize_embeddings=True)
                embeddings.extend(batch_embeddings)
            except Exception as e:
                # loop individually to get embeddings within batch
                for text in batch:
                    try:
                        single_embeddings = model.encode([text], normalize_embeddings=True)[0]
                        embeddings.append(single_embeddings)
                    except Exception as retry_err:
                        # if error in processing embeddings within batch, put in a placeholder
                        placeholder_embedding = np.zeros(MODEL_DIMS["e5"])
                        embeddings.append(placeholder_embedding)
        self.node_feat_e5 = torch.tensor(embeddings, dtype=torch.float32).to(device)
        if return_val:
            return self.node_feat_e5.cpu().numpy()

    def prepare_data(self, return_val = True):
        self.load_model()
        self.get_nodeidx_mappings(return_val=False)
        self.generate_e5_embeddings(return_val=False)
        self.graph = self.graph.to(self._args.device)
        if self._args.dataset == 'ogbn-arxiv':
            self.graph.ndata['e5_feat'] = self.node_feat_e5
            self.graph.ndata['paper_id'] = self.paperid
        elif self._args.dataset == 'ogbn-mag':
            self.graph.nodes['paper'].data['e5_feat'] = self.node_feat_e5
            self.graph.nodes['paper'].data['paper_id'] = self.paperid
        elif self._args.dataset == 'combined':
            self.graph.ndata['e5_feat'] = self.node_feat_e5
            self.graph.ndata['paper_id'] = self.paperid

        if return_val:
            return self.node_feat_e5.cpu().numpy()
        #Here "e5_feat" is the normal embedding, and later generated graphalign embedding is saved to self.graph.ndata['ga_embedding']
        #We can load the datasetName_graph.bin to get the normal embedding.

    def infer_graphalign(self, return_val = True):
        print("Generating GraphAlign embeddings...")
        self.model.eval()
        batch_size = self._args.batch_size_f
        with torch.no_grad():
            #init features
            all_embeddings = []

            # prepare data
            if self._args.dataset == 'ogbn-arxiv':
                x = self.graph.ndata['e5_feat'].clone().to(self._device)
                num_nodes = len(x)
                g = self.graph
            elif self._args.dataset == 'ogbn-mag':
                x = self.graph.nodes['paper'].data['e5_feat'].clone().to(self._device)
                paper_edges = ('paper', 'cites', 'paper')
                num_nodes = len(x)
                g = self.graph.edge_type_subgraph([paper_edges])
            elif self._args.dataset == 'combined':
                x = self.graph.ndata['e5_feat'].clone().to(self._device)
                num_nodes = len(x)
                g = self.graph

            # Process in batches
            for start_idx in tqdm(range(0, num_nodes, batch_size), desc="Processing batches"):
                end_idx = min(start_idx + batch_size, num_nodes)
                batch_nodes = torch.arange(start_idx, end_idx).to(self._device)
                batch_emb = self.model.embed(g, x)[batch_nodes]
                all_embeddings.append(batch_emb.cpu())
            torch_embedding = torch.cat(all_embeddings, dim=0).to(self._device)

        # add embedding to graph
        self.graph = self.graph.to(self._args.device)
        if self._args.dataset == 'ogbn-arxiv' or 'combined':
            self.graph.ndata['ga_embedding'] = torch_embedding
        elif self._args.dataset == 'ogbn-mag':
            self.graph.nodes['paper'].data['ga_embedding'] = torch_embedding
        if return_val:
            return torch_embedding.cpu().numpy()

    def prepare_dataset(self):
        split_idx = self.dataset.get_idx_split()
        self.train_idx, self.valid_idx, self.test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]

def data_loading_thread(data_queue, dataloader):
    for batch in dataloader:
        data_queue.put(batch)

def open_pickle(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def process_papers(dataset_name):
    # load Paper.txt
    paper_path = os.path.join(data_dir, 'Papers.txt')
    column_names = ['paper_id', 'Rank', 'Doi', 'DocType', 'PaperTitle', 'OriginalTitle',
                        'BookTitle','Year', 'Date', 'OnlineDate', 'Publisher', 'JournalId',
                        'ConferenceSeriesId','ConferenceInstanceId', 'Volume', 'Issue',
                        'FirstPage', 'LastPage','ReferenceCount', 'CitationCount',
                        'EstimatedCitation', 'OriginalVenue','FamilyId', 'FamilyRank', 'DocSubTypes', 'CreatedDate'
    ]
    df = pd.read_csv(
        paper_path,
        sep='\t',
        header=None,
        names=column_names,
        on_bad_lines='skip',
        quoting=3
    )

    if dataset_name == 'ogbn-mag':
        # load paper_entidx2name.csv.gz
        node_idx = pd.read_csv(
            os.path.join(data_dir, 'paper_entidx2name.csv.gz'),
            sep=',',
            compression='gzip',
            header=0,
            names=['node_idx', 'paper_id']
        )
        node_idx['paper_id'] = node_idx['paper_id'].astype(np.int64)

        # left join
        node_titles = node_idx.merge(df.reset_index(), left_on='paper_id', right_on='paper_id', how='left')

        #filter columns
        node_titles['OriginalTitle'] = node_titles['OriginalTitle'].fillna('')
        node_titles['Year'] = node_titles['Year'].fillna(0)
        node_titles['Year'] = node_titles['Year'].astype(int)

        # save
        node_titles.to_csv(os.path.join(data_dir, 'mag-nodeidx2titles.csv'), index=False)
        return node_titles
    elif dataset_name == 'combined':
        graph_path = os.path.join(data_dir,'combined_graph_paper_id.pkl.gz')
        with gzip.open(graph_path, 'rb') as f:
            graph = pickle.load(f)
        paper_id = pd.DataFrame(graph.ndata['paper_id'].cpu().numpy(), columns=['paper_id'])
        paper_id['paper_id'] = paper_id['paper_id'].astype(np.int64)

        # left join
        node_titles = paper_id.merge(df.reset_index(), left_on='paper_id', right_on='paper_id', how='left')

         #filter columns
        node_titles['OriginalTitle'] = node_titles['OriginalTitle'].fillna('')
        node_titles['Year'] = node_titles['Year'].fillna(0)
        node_titles['Year'] = node_titles['Year'].astype(int)

        # save
        node_titles.to_csv(os.path.join(data_dir, 'combined-nodeidx2titles.csv'), index=False)
        return node_titles


def pulse_check():
        print("Process is still running...")
