
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
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import gensim.downloader as api
from src.trainer_large import ModelTrainer, build_model
from src.utils.utils import set_random_seed

# Global variables
dataset_names = ['ogbn-mag','ogbn-arxiv','ogbn-products','combined']
utils_path = os.path.dirname(os.path.abspath(__file__))
cwd = os.getcwd()
model_dir = os.path.join(cwd, 'model')
data_dir = os.path.join(cwd, 'data')
model_path = os.path.join(model_dir, 'GraphAlign_graphmae.pt')
stark_path = os.path.join(data_dir, 'stark-mag')

graph_path = os.path.join(data_dir,'combined_graph_paper_id.pkl.gz')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


with gzip.open(graph_path, 'rb') as f:
    graph = pickle.load(f)
    graph = graph.to(device)
    label = graph.ndata['label']
    train_idx = np.where(graph.ndata['train_mask'] == 1)[0]
    val_idx = np.where(graph.ndata['valid_mask'] == 1)[0]
    test_idx = np.where(graph.ndata['test_mask'] == 1)[0]
    feat_type = 'word2vec'

idx_path = os.path.join(data_dir, 'combined-nodeidx2titles.csv')
df = pd.read_csv(
                idx_path,
                sep=',',
                header= 0
            )


node_titles = df[['paper_id', 'OriginalTitle']]
node_titles.rename(columns={'OriginalTitle': 'title'}, inplace=True)
paperid = torch.tensor(graph.ndata['paper_id'],dtype=torch.int64).to(device)


def get_sentence_embedding(sentence, model):
    words = sentence.lower().split()
    word_embeddings = [model[word] for word in words if word in model]
    if not word_embeddings:
        return np.zeros(model.vector_size)
    sentence_embedding = np.mean(word_embeddings, axis=0)
    return sentence_embedding

model = api.load("word2vec-google-news-300")

embeddings = []
for text in tqdm(node_titles['title']):
    # Get the sentence embedding
    embedding = get_sentence_embedding(str(text), model)
    embeddings.append(embedding)

#change to tensor array
embeddings = torch.tensor(embeddings, dtype=torch.float32).to(device)
graph.ndata['w2v'] = embeddings


def process_feature(graph, feature_name, n_components):
    scaler = StandardScaler()
    scaled_features = torch.tensor(scaler.fit_transform(graph.ndata[feature_name].cpu().numpy())).float()
    print(f"Feature '{feature_name}' normalized. Mean: {scaled_features.mean():.4f}, Std: {scaled_features.std():.4f}")
    pca = PCA(n_components=n_components)
    reduced_features = torch.tensor(pca.fit_transform(scaled_features.numpy())).float()
    graph.ndata[feature_name] = reduced_features
    explained_variance = sum(pca.explained_variance_ratio_) * 100
    print(f"Feature '{feature_name}' reduced to {n_components} dimensions. Explained variance: {explained_variance:.2f}%")
    return graph

# run pca
graph = graph.cpu()
graph_p = process_feature(graph, 'w2v', 128)

# save graph to .bin file
from dgl.data.utils import save_graphs
processed_dir = os.path.join(cwd, 'processed')
save_graphs(os.path.join(processed_dir, 'pca_graph.bin'), [graph_p], labels={'glabel': graph_p.ndata['label']})


