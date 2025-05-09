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
# set params
env = 'terminal'
if env=='terminal':
    cwd = os.getcwd()
else:
    cwd = os.path.dirname(__file__)

# device torch
device = th.device('cuda' if th.cuda.is_available() else 'cpu')
th.manual_seed(42)

# set foldrs
data_dir = os.path.join(cwd, 'data')
processed_dir = os.path.join(cwd, 'processed')

# load graph path
GRAPHS = ['combined_graph_pca.bin','graph0.bin','graph1.bin','pca_graph.bin']
if args.graph_idx == 0:
   graph_name = 'joiened'
elif args.graph_idx == 1:
   graph_name = 'MAG'
elif args.graph_idx == 2:
   graph_name = 'arxiv'
elif args.graph_idx == 3:
   graph_name = 'pca'
graph_path = os.path.join(processed_dir,GRAPHS[args.graph_idx])

# load graph
graphs, _ = load_graphs(graph_path)
graph = graphs[0]
labels = graph.ndata['label']


#load features
if graph_name == 'pca':
   pca_feature = graph.ndata['w2v']
else:
   e5_features = graph.ndata['e5_feat']
   ga_features = graph.ndata['ga_embedding']
   original_features = graph.ndata['feat']


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

# create model
gat_e5 = GAT(e5_features.shape[1],e5_hidden_dim , output_dim, gat_heads)
gat_ga = GAT(ga_features.shape[1],ga_hidden_dim , output_dim, gat_heads)
gat_original = GAT(original_features.shape[1],original_hidden_dim , output_dim, gat_heads)


# Move models and data `to device
gat_e5 = gat_e5.to(device)
gat_ga = gat_ga.to(device)
gat_original = gat_original.to(device)
graph = graph.to(device)
e5_features = e5_features.to(device)
ga_features = ga_features.to(device)
original_features = original_features.to(device)
labels = labels.to(device)


# train model with normal sampling
to_sample = args.sample

print("Model Architecture of e5: ")
print(gat_e5)
gat_e5_state, gat_e5_best_train, gat_e5_train_f1, gat_e5_best_val, gat_e5_best_f1,gat_e5_best_epoch = train_model(
   gat_e5,
   graph,
   e5_features,
   labels,
   train_idx,
   val_idx,
   epochs=100,
   lr=0.005,
   to_sample=to_sample
)
gat_e5.load_state_dict(gat_e5_state)
print("------------------------------------------------")

print("Model Architecture of graphalign: ")
print(gat_ga)
gat_ga_state, gat_ga_best_train, gat_ga_train_f1, gat_ga_best_val,gat_ga_best_f1, gat_ga_best_epoch= train_model(
   gat_ga,
   graph,
   ga_features,
   labels,
   train_idx,
   val_idx,
   epochs=100,
   lr=0.005,
   to_sample=to_sample
)
gat_ga.load_state_dict(gat_ga_state)
print("------------------------------------------------")


print("Model Architecture of original: ")
print(gat_original)
gat_original_state, gat_original_best_train, gat_original_train_f1, gat_original_best_val,gat_original_best_f1, gat_original_best_epoch = train_model(
   gat_original,
   graph,
   original_features,
   labels,
   train_idx,
   val_idx,
   epochs=100,
   lr=0.005,
   to_sample=to_sample
)
gat_original.load_state_dict(gat_original_state)
print("------------------------------------------------")

# evaluate
gat_e5_test_accuracy, gat_e5_test_f1 = evaluate(
   gat_e5,
   graph,
   e5_features,
   labels,
   graph.ndata['test_mask']
)
gat_ga_test_accuracy, gat_ga_test_f1 = evaluate(
   gat_ga,
   graph,
   ga_features,
   labels,
   graph.ndata['test_mask']
)
gat_original_test_accuracy, gat_original_test_f1 = evaluate(
   gat_original,
   graph,
   original_features,
   labels,
   graph.ndata['test_mask']
)

# print results
print(f"GAT E5 Best Validation Accuracy: {gat_e5_best_val:.4f}, F1: {gat_e5_best_f1:.4f}")
print(f"GAT E5 Test Accuracy: {gat_e5_test_accuracy:.4f}, F1: {gat_e5_test_f1:.4f}")
print(f"GAT GA Best Validation Accuracy: {gat_ga_best_val:.4f}, F1: {gat_ga_best_f1:.4f}")
print(f"GAT GA Test Accuracy: {gat_ga_test_accuracy:.4f}, F1: {gat_ga_test_f1:.4f}")
print(f"GAT Original Best Validation Accuracy: {gat_original_best_val:.4f}, F1: {gat_original_best_f1:.4f}")
print(f"GAT Original Test Accuracy: {gat_original_test_accuracy:.4f}, F1: {gat_original_test_f1:.4f}")

# save results
results = {
   'gat_e5_train_accuracy': gat_e5_best_train,
   'gat_e5_train_f1': gat_e5_train_f1,
   'gat_e5_test_accuracy': gat_e5_test_accuracy,
   'gat_e5_test_f1': gat_e5_test_f1,
   'gat_e5_best_val': gat_e5_best_val,
   'gat_e5_best_f1': gat_e5_best_f1,
   'gat_ga_train_accuracy': gat_ga_best_train,
   'gat_ga_train_f1': gat_ga_train_f1,
   'gat_ga_test_accuracy': gat_ga_test_accuracy,
   'gat_ga_test_f1': gat_ga_test_f1,
   'gat_ga_best_val': gat_ga_best_val,
   'gat_ga_best_f1': gat_ga_best_f1,
   'gat_original_train_accuracy': gat_original_best_train,
   'gat_original_train_f1': gat_original_train_f1,
   'gat_original_test_accuracy': gat_original_test_accuracy,
   'gat_original_test_f1': gat_original_test_f1,
   'gat_original_best_val': gat_original_best_val,
   'gat_original_best_f1': gat_original_best_f1
}

df = [{
   'Model' : 'GAT',
   'Graph' : graph_name,
   'Feature': 'E5 Embedding',
   'Best Val Epoch': gat_e5_best_epoch,
   'Train_Accuracy': gat_e5_best_train,
   'Train_F1': gat_e5_train_f1,
   'Best_Val_Accuracy': gat_e5_best_val,
   'Best_Val_F1': gat_e5_best_f1,
   'Test_Accuracy': gat_e5_test_accuracy,
   'Test_F1': gat_e5_test_f1,
},
{
   'Model' : 'GAT',
   'Graph' : graph_name,
   'Feature': 'GraphAlign',
   'Best Val Epoch': gat_ga_best_epoch,
   'Train_Accuracy': gat_ga_best_train,
   'Train_F1': gat_ga_train_f1,
   'Best_Val_Accuracy': gat_ga_best_val,
   'Best_Val_F1': gat_ga_best_f1,
   'Test_Accuracy': gat_ga_test_accuracy,
   'Test_F1': gat_ga_test_f1,
},
{
   'Model' : 'GAT',
   'Graph' : graph_name,
   'Feature': 'Original',
   'Best Val Epoch': gat_e5_best_epoch,
   'Train_Accuracy': gat_original_best_train,
   'Train_F1': gat_original_train_f1,
   'Best_Val_Accuracy': gat_original_best_val,
   'Best_Val_F1': gat_original_best_f1,
   'Test_Accuracy': gat_original_test_accuracy,
   'Test_F1': gat_original_test_f1,
}
]


# save results
if to_sample:
   sampled = 'multihop'
else:
   sampled = 'batchify'

with open(os.path.join(processed_dir, f'gat_results_{graph_name}_{sampled}.pkl'), 'wb') as f:
   pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

#save csv
save_df = pd.DataFrame(df)
save_df.to_csv(os.path.join(processed_dir, f'gat_results_{graph_name}_{sampled}.csv'), index=True)

# save model
#th.save(gat_e5.state_dict(), os.path.join(processed_dir, 'gat_e5_model.pth'))
#th.save(gat_ga.state_dict(), os.path.join(processed_dir, 'gat_ga_model.pth'))
#th.save(gat_original.state_dict(), os.path.join(processed_dir, 'gat_original_model.pth'))

