import torch
import torch.multiprocessing
import dgl.dataloading
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
def preprocess(graph):
    graph = dgl.to_bidirected(graph)
    graph = graph.remove_self_loop().add_self_loop()
    return graph


def scale_feats(x):
    scaler = StandardScaler()
    feats = x.numpy()
    scaler.fit(feats)
    feats = torch.from_numpy(scaler.transform(feats)).float()
    return feats

def multi_scale_feats(x):
    scaler = StandardScaler()
    scaler.fit(get_all_embedding())
    feats = x.numpy()
    feats = torch.from_numpy(scaler.transform(feats)).float()
    return feats


          