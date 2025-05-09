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
from dgl.data.utils import load_graphs
from torchsummary import summary
import pandas as pd

from utils.model import GAT, MLP, evaluate, train_model
import argparse

#get arguments
parser = argparse.ArgumentParser(
                    prog='MSBD5008 Graph Embedding Training',
                    description='Graph embedding')
parser.add_argument('--sample',  default=False,action='store_true',
                    help='True: multihop sampling, False: batchify sampling')
parser.add_argument('--graph_idx', default=0, type=int,)
args = parser.parse_args()

#init dir
cwd = os.getcwd()
device = th.device('cuda' if th.cuda.is_available() else 'cpu')
th.manual_seed(42)

# set foldrs
data_dir = os.path.join(cwd, 'data')
processed_dir = os.path.join(cwd, 'processed')

GRAPHS = ['combined_graph_pca.bin','graph0.bin','graph1.bin','pca_graph.bin']
graph_idx = 3
graph_path = os.path.join(processed_dir,GRAPHS[graph_idx])
graph_name = 'pca'
pca_features = graph.ndata['w2v']
graphs, _ = load_graphs(graph_path)
graph = graphs[0]
labels = graph.ndata['label']
if graph_name == 'pca':
   graph.ndata['feat'] = graph.ndata['w2v']


#get indices
train_idx = np.where(graph.ndata['train_mask'] == 1)[0]
val_idx = np.where(graph.ndata['valid_mask'] == 1)[0]
test_idx = np.where(graph.ndata['test_mask'] == 1)[0]

# init net
graph = dgl.add_self_loop(graph)
output_dim = labels.shape[1]

#adjust architecture by embedding type
# node2vec = 128
# e5_features = 384
# ga_features = 1024
gat_heads = 1
original_hidden_dim = 512
e5_hidden_dim = 512
ga_hidden_dim = 512
batch_size = 1024

gat_pca =  GAT(pca_features.shape[1],e5_hidden_dim , output_dim, gat_heads)
graph = graph.to(device)
e5_features = pca_features.to(device)
labels = labels.to(device)

to_sample = False

print("Model Architecture of e5: ")
print(gat_pca)

gat_pca_state, gat_pca_best_train, gat_pca_train_f1, gat_pca_best_val, gat_pca_best_f1,gat_pca_best_epoch = train_model(
   gat_pca,
   graph,
   e5_features,
   labels,
   train_idx,
   val_idx,
   epochs=100,
   lr=0.005,
   to_sample=to_sample
)
gat_pca.load_state_dict(gat_pca_state)
print("------------------------------------------------")

gat_pca_test_accuracy, gat_pca_test_f1 = evaluate(
   gat_pca,
   graph,
   e5_features,
   labels,
   graph.ndata['test_mask']
)


print(f"Best Train Accuracy: {gat_pca_best_train}")
print(f"Best Train F1 Score: {gat_pca_train_f1}")
print(f"Best Val Accuracy: {gat_pca_best_val}")
print(f"Best Val F1 Score: {gat_pca_best_f1}")
print(f"Test Accuracy: {gat_pca_test_accuracy}")
print(f"Test F1 Score: {gat_pca_test_f1}")
print(f"Best Epoch: {gat_pca_best_epoch}")

#save model
th.save(gat_pca.state_dict(), os.path.join(processed_dir, 'gat_pca_model.pth'))
