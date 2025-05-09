import os
import pickle
from argparse import Namespace

import numpy as np
import pandas as pd
import torch
from dgl.data.utils import save_graphs
from dgl.dataloading import GraphDataLoader

import utils.data_utils as du
from utils.data_utils import GraphAlign_e5, load_ogb_dataset, open_pickle

# set params
env = 'terminal'
if env=='terminal':
    cwd = os.getcwd()
else:
    cwd = os.path.dirname(__file__)


# set foldrs
model_path = os.path.join(cwd, 'model', 'GraphAlign_graphmae.pt')
config_path = os.path.join(cwd, 'src', 'config', 'GraphMAE_configs.yml')
data_dir = os.path.join(cwd, 'data')

# cuda device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# datset names
dataset_names = ['ogbn-mag','ogbn-arxiv','combined','pca-word2vec']
idx_dataset = input("Select dataset (0: ogbn-mag, 1: ogbn-arxiv, 2: combined, 3: pca_w2v): ")
idx_dataset = int(idx_dataset)  # Convert to integer
if idx_dataset < 0 or idx_dataset >= len(dataset_names):
    raise ValueError("Invalid dataset index. Please choose 0 or 1.")

# set arguments
args = Namespace(
    dataset=dataset_names[idx_dataset],
    model='graphmae',
    use_cfg=True,
    device=0,
    load_model=True,
    load_model_path=model_path,
    data_dir=data_dir,
    feat_type='e5_float16',
    use_cfg_path=config_path,
    prob_num_workers=4,
    moe=True,
    moe_use_linear=True,
    top_k=1,
    num_expert=4,
    hiddenhidden_size_times=1,
    moe_layer=[0],
    decoder_no_moe=True,
    num_hidden=1024,
    num_layers=4,
    num_proj_hidden=1024,
    num_heads=4,
    encoder='gat',
    activation='prelu',
    norm='layernorm',
    num_out_heads=1,
    in_drop=0.2,
    attn_drop=0.2,
    feat_drop=0.2,
    negative_slope=0.2,
    residual=True,
    mask_rate=0.5,
    replace_rate=0.0,
    alpha_l=2,
    weight_decay_f=1e-4,
    lr_f=0.001,
    max_epoch_f=1000,
    batch_size_f=10000,
    scheduler=True,
    loss_fn='sce',
    optimizer='adamw',
    linear_prob=True,
    dropout=0.2,
    decoder='gat',
    drop_edge_rate=0.0,
    concat_hidden=False,
    deepspeed=False,
    lr=0.001,
    weight_decay=0.0001,
    num_workers=4,
    num_epochs=1000,
    no_verbose=False,
    save_model=True
)

# create & load model
model = GraphAlign_e5(args)
e5_embedding = model.prepare_data()

# infer graph alignment embeddings
ga_embeddings = model.infer_graphalign()

# save generated embeddings
dataset_name = dataset_names[idx_dataset]
processed_dir = os.path.join(cwd, 'processed')
if not os.path.exists(processed_dir):
    os.makedirs(processed_dir)

#save graph model
if dataset_names[idx_dataset]== 'ogbn-arxiv':
    save_graphs(os.path.join(cwd, 'processed', f'{dataset_name}_graph.bin'), model.graph, labels = {'glabel':model.label})
elif dataset_names[idx_dataset]== 'ogbn-mag':
    save_graphs(os.path.join(cwd, 'processed', f'{dataset_name}_graph.bin'), model.graph, labels = {'glabel':model.label['paper']})
elif dataset_names[idx_dataset]== 'combined':
    save_graphs(os.path.join(cwd, 'processed', f'{dataset_name}_graph.bin'), model.graph, labels = {'glabel':model.label})

# save graphalign embeddings pt1
torch.save(
    e5_embedding,
    os.path.join(processed_dir, f'{dataset_name}_e5_embeddings.pt'),
    pickle_protocol=4  # Specify protocol 4
)

torch.save(
    ga_embeddings,
    os.path.join(processed_dir, f'{dataset_name}_graphalign_embeddings.pt'),
    pickle_protocol=4  # Specify protocol 4
)


# save the paper-id to embedding
if dataset_names[idx_dataset] == 'ogbn-arxiv':
    paper_id = model.graph.ndata['paper_id']
    paper_id = paper_id.cpu().numpy()
    mappings = {
        'paper_id': paper_id,
        'e5_embedding': e5_embedding.numpy(),
        'ga_embedding': ga_embeddings
    }
elif dataset_names[idx_dataset] == 'ogbn-mag':
    paper_id = model.graph.nodes['paper'].data['paper_id']
    paper_id = paper_id.cpu().numpy()
    mappings = {
            'paper_id': paper_id,
            'e5_embedding': e5_embedding,
            'ga_embedding': ga_embeddings
    }
elif dataset_names[idx_dataset] == 'combined':
    paper_id = model.graph.ndata['paper_id']
    paper_id = paper_id.cpu().numpy()
    mappings = {
        'paper_id': paper_id,
        'e5_embedding': e5_embedding,
        'ga_embedding': ga_embeddings
    }

# save pickle
with open(os.path.join(processed_dir, f'{dataset_name}_mappings.pkl'), 'wb') as f:
    pickle.dump(mappings, f, protocol=pickle.HIGHEST_PROTOCOL)

