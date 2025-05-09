
#load the graph
import os
import pickle
import gzip
import networkx as nx
import numpy as np

#dir
cwd = os.getcwd()
data_dir = os.path.join(cwd, 'data')
import_graph_path = os.path.join(data_dir,'combined_graph_with3embedding_processed.pkl.gz')
processed_dir = os.path.join(cwd, 'processed')


#load graph
with gzip.open(import_graph_path, 'rb') as f:
    g = pickle.load(f)


# save graph
from dgl import save_graphs
graph_path = os.path.join(processed_dir,'combined_graph_pca.bin')
save_graphs(graph_path, g)
