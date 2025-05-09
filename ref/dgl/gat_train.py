import gzip
import os
import pickle

import dgl
import numpy as np
import torch as th
from dgl.data.utils import load_graphs
from dgl.nn import GATConv
import torch.nn.functional as F
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from dgl.dataloading import GraphDataLoader

from utils.model import GAT, MLP, evaluate, train_model

#init dir
# set params
env = 'terminal'
if env=='terminal':
    cwd = os.getcwd()
else:
    cwd = os.path.dirname(__file__)

# device torch
device = th.device('cuda' if th.cuda.is_available() else 'cpu')

# set foldrs
data_dir = os.path.join(cwd, 'data')
processed_dir = os.path.join(cwd, 'processed')
graph_path = os.path.join(data_dir,'combined_graph_paper_id.pkl.gz')
arxiv_mapping_path = os.path.join(processed_dir, 'ogbn-arxiv_mappings.pkl')
mag_mapping_path = os.path.join(processed_dir, 'ogbn-mag_mappings.pkl')

# load data
with gzip.open(graph_path, 'rb') as f:
  graph = pickle.load(f)

#load arxiv
with open(arxiv_mapping_path,'rb') as f:
  arxiv_data = pickle.load(f)

#load mag
with open(mag_mapping_path,'rb') as f:
  mag_data = pickle.load(f)

# init variables
paper_id = graph.ndata['paper_id']
mag_paper_id = mag_data['paper_id']
mag_e5_embedding = mag_data['e5_embedding']
mag_ga_embedding = mag_data['ga_embedding']
arxiv_paper_id = arxiv_data['paper_id']
arxiv_e5_embedding = arxiv_data['e5_embedding']
arxiv_ga_embedding = arxiv_data['ga_embedding']
graph_source = graph.ndata['graph_source'].numpy()
arxiv_mask = graph_source == 0
mag_mask = graph_source != 0

# Fill dictionaries with paper_id -> embedding mappings
arxiv_id_to_embedding = {}
mag_id_to_embedding = {}
for i, paper_id in enumerate(arxiv_paper_id):
    arxiv_id_to_embedding[paper_id] = {
        'e5': arxiv_e5_embedding[i],
        'ga': arxiv_ga_embedding[i]
    }

for i, paper_id in enumerate(mag_paper_id):
    mag_id_to_embedding[paper_id] = {
        'e5': mag_e5_embedding[i],
        'ga': mag_ga_embedding[i]
    }

# Initialize embedding tensors for the combined graph
e5_embeddings = []
ga_embeddings = []

# Collect embeddings in the order of nodes in the combined graph
for i in range(graph.num_nodes()):
  paper_id = graph.ndata['paper_id'][i].item()  # Get paper_id for this node
  if paper_id in mag_id_to_embedding:
    e5_embeddings.append(mag_id_to_embedding[paper_id]['e5'])
    ga_embeddings.append(mag_id_to_embedding[paper_id]['ga'])
  elif paper_id in arxiv_id_to_embedding:
    e5_embeddings.append(arxiv_id_to_embedding[paper_id]['e5'])
    ga_embeddings.append(arxiv_id_to_embedding[paper_id]['ga'])
  else:
    print(f"Warning: ArXiv paper_id {paper_id} not found in mappings")
    # Use zeros as fallback
    e5_embeddings.append(np.zeros_like(arxiv_e5_embedding[0]))
    ga_embeddings.append(np.zeros_like(arxiv_ga_embedding[0]))

graph.ndata['e5'] = th.tensor(np.array(e5_embeddings))
graph.ndata['graphalign'] = th.tensor(np.array(ga_embeddings))
e5_features = graph.ndata['e5']
ga_features = graph.ndata['graphalign']
original_features = graph.ndata['feat']  # The original features from the dataset
labels = graph.ndata['label']

#get indices
train_idx = np.where(graph.ndata['train_mask'] == 1)[0]
val_idx = np.where(graph.ndata['valid_mask'] == 1)[0]
test_idx = np.where(graph.ndata['test_mask'] == 1)[0]

# init net
graph = dgl.add_self_loop(graph)
input_dim = graph.ndata['feat'].shape[1]
output_dim = labels.shape[1]
gat_heads = 8
hidden_dim = 256
batch_size = 2056


# create model
gat_e5 = GAT(e5_features.shape[1],hidden_dim , output_dim, gat_heads)
gat_ga = GAT(ga_features.shape[1],hidden_dim , output_dim, gat_heads)
gat_original = GAT(original_features.shape[1],hidden_dim , output_dim, gat_heads)

# Move models and data to device
gat_e5 = gat_e5.to(device)
gat_ga = gat_ga.to(device)
gat_original = gat_original.to(device)
graph = graph.to(device)
e5_features = e5_features.to(device)
ga_features = ga_features.to(device)
original_features = original_features.to(device)
labels = labels.to(device)


# train modelx
gat_e5_state, gat_e5_best_val = train_model(
   gat_e5,
   graph,
   e5_features,
   labels,
   train_idx,
   val_idx,
   epochs=200,
   lr=0.005
)
gat_e5.load_state_dict(gat_e5_state)

gat_ga_state, gat_ga_best_val = train_model(
   gat_ga,
   graph,
   ga_features,
   labels,
   train_idx,
   val_idx,
   epochs=200,
   lr=0.005
)
gat_ga.load_state_dict(gat_ga_state)

gat_original_state, gat_original_best_val = train_model(
   gat_original,
   graph,
   original_features,
   labels,
   train_idx,
   val_idx,
   epochs=200,
   lr=0.005
)
gat_original.load_state_dict(gat_original_state)

# evaluate
gat_e5_accuracy, gat_e5_f1 = evaluate(
   gat_e5,
   graph,
   e5_features,
   labels,
   graph.ndata['test_mask']
)
gat_ga_accuracy, gat_ga_f1 = evaluate(
   gat_ga,
   graph,
   ga_features,
   labels,
   graph.ndata['test_mask']
)
gat_original_accuracy, gat_original_f1 = evaluate(
   gat_original,
   graph,
   original_features,
   labels,
   graph.ndata['test_mask']
)

# print results
print(f"GAT E5 Accuracy: {gat_e5_accuracy:.4f}, F1: {gat_e5_f1:.4f}")
print(f"GAT GA Accuracy: {gat_ga_accuracy:.4f}, F1: {gat_ga_f1:.4f}")
print(f"GAT Original Accuracy: {gat_original_accuracy:.4f}, F1: {gat_original_f1:.4f}")

