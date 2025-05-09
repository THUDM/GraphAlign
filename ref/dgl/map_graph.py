import gzip
import os
import pickle

import dgl
import numpy as np
import torch as th
from dgl.data.utils import load_graphs
from dgl.nn import GATConv

# set params
env = 'terminal'
if env=='terminal':
    cwd = os.getcwd()
else:
    cwd = os.path.dirname(__file__)

#get folders
processed_path = os.path.join(cwd, 'processed')
data_dir = os.path.join(cwd, 'data')
model_dir = os.path.join(cwd, 'model')

# load paper id mappings
arxiv_mapping = os.path.join(processed_path, 'ogbn-arxiv_mappings.pkl')
with open(arxiv_mapping, 'rb') as f:
    arxiv_mappings = pickle.load(f)
arxiv_paper_id = arxiv_mappings['paper_id']
arxiv_e5_embedding = arxiv_mappings['e5_embedding']
arxiv_ga_embedding = arxiv_mappings['ga_embedding']

mag_mapping = os.path.join(processed_path, 'ogbn-mag_mappings.pkl')
with open(mag_mapping, 'rb') as f:
    mag_mappings = pickle.load(f)
mag_paper_id = mag_mappings['paper_id']
mag_e5_embedding = mag_mappings['e5_embedding']
mag_ga_embedding = mag_mappings['ga_embedding']

#load pkl.gz file
graph_path = os.path.join(data_dir, 'combined_graph.pkl.gz')
with gzip.open(graph_path, 'rb') as f:
    graph = pickle.load(f)

# get '_ID','graph_source'
paper_id = graph.ndata['paper_id']
graph_source = graph.ndata['graph_source'].numpy()
arxiv_mask = graph_source == 0
mag_mask = graph_source != 0

# get a map of all e5 and ga_embeddings based on index
e5_mapping = []
ga_mapping = []
for i in range(graph.num_nodes()):
    if arxiv_mask[i]:
        node_id = graph_id[i] 
        e5_mapping.append(arxiv_e5_embedding[node_id])
        ga_mapping.append(arxiv_ga_embedding[node_id])
    else:
        node_id = graph_id[i]
        e5_mapping.append(mag_e5_embedding[node_id])
        ga_mapping.append(mag_ga_embedding[node_id])


# assign e5 & ga embedding for each node in the graph
for i in range(graph.num_nodes()):
    paper_id = graph.ndata['_ID'].numpy()[i] #the name here changed to node_id is better
    graph.ndata['e5'][i] = th.tensor(e5_mapping[paper_id]) #same change
    graph.ndata['graphalign'][i] = th.tensor(ga_mapping[paper_id]) #same change

# save graph with e5 & ga embeddings
dgl_save_path = os.path.join(processed_path, 'combined_graph_with_embeddings.bin')




