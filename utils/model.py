
import dgl
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl.dataloading import DataLoader, MultiLayerFullNeighborSampler
from dgl.nn import GATConv
from sklearn.metrics import accuracy_score, f1_score

# device torch
device = th.device('cuda' if th.cuda.is_available() else 'cpu')
th.manual_seed(42)

to_sample = True

class GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads):
        super(GAT, self).__init__()
        self.layer1 = GATConv(in_dim, hidden_dim, num_heads=num_heads,attn_drop = 0.3,feat_drop=0.3,activation=F.relu)
        self.layer2 = GATConv(hidden_dim * num_heads, hidden_dim, num_heads=num_heads, attn_drop = 0.3,feat_drop=0.3,activation=F.relu)
        self.layer3 = GATConv(hidden_dim, out_dim, num_heads=1, attn_drop = 0.3,feat_drop=0.3,activation=F.relu)

    def forward(self, graph, h):
        # Handle both full graph and block-based inputs
        if isinstance(graph, list):  # If we're using blocks
            h = self.layer1(graph[0], h)
            h = h.view(h.shape[0], -1)

            # layer2
            h = self.layer2(graph[1], h)
            h = h.view(h.shape[0], -1)

            # layer3
            h = self.layer3(graph[2], h)
            h = h.view(h.shape[0], -1)
            h = h.squeeze(1)
        else:
            h = self.layer1(graph, h)
            h = h.view(h.shape[0], -1)

            h = self.layer2(graph, h)
            h = h.view(h.shape[0], -1)

            h = self.layer3(graph, h)
            h = h.view(h.shape[0], -1)
            h = h.squeeze(1)
        return h

### old implementation  ignore ###
class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)

    def forward(self, h):
        h = F.dropout(h, p=0.2, training=self.training)
        h = F.relu(self.fc1(h))
        h = F.dropout(h, p=0.2, training=self.training)
        h = F.relu(self.fc2(h))
        h = self.fc3(h)
        return h
### old implementation  ignore ###

# Function to evaluate model
def evaluate(model, graph, features, labels, mask, is_mlp=False):
    model.eval()
    with th.no_grad():
        if is_mlp:
            logits = model(features)
        else:
            logits = model(graph, features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = th.max(logits, dim=1)

        # convert
        indices = indices.cpu().numpy()
        labels = labels.cpu().numpy()
        labels = np.argmax(labels,axis=1)

        accuracy = accuracy_score(labels, indices)
        f1 = f1_score(labels, indices, average='macro')
    return accuracy, f1

# Function to train model
def train_model(model, graph, features, labels, train_idx, val_idx, epochs=200, lr=0.005, is_mlp=False, batch_size=2056, to_sample=True):
    optimizer = th.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    best_val_acc = 0
    best_f1 = 0
    best_train_acc = 0
    best_train_f1 = 0
    best_model_state = None
    model.training = True

    # Convert indices to tensors if they're not already
    if isinstance(train_idx, np.ndarray):
        train_idx = th.from_numpy(train_idx)
    if isinstance(val_idx, np.ndarray):
        val_idx = th.from_numpy(val_idx)

    # Move model and data to device
    model = model.to(device)
    graph = graph.to(device)
    features = features.to(device)
    labels = labels.to(device)
    train_idx = train_idx.to(device)
    val_idx = val_idx.to(device)

    # Set up data loader
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(3)
    train_loader = DataLoader(
        graph, train_idx, sampler,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0
    )

    val_loader = DataLoader(
        graph, val_idx, sampler,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0
    )

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        batch_count = 0

        if to_sample == False:
            # Training with batches - only on train_idx
            for input_nodes, output_nodes, blocks in train_loader:
                batch_count += 1

                # Get the features and labels for this batch
                batch_inputs = features[input_nodes]
                batch_labels = labels[output_nodes].float()
                batch_inputs = batch_inputs.to(device)
                batch_labels = batch_labels.to(device)
                blocks = [block.to(device) for block in blocks]
                input_nodes = input_nodes.to(device)
                output_nodes = output_nodes.to(device)

                # Forward pass - different handling for MLP and graph models
                if is_mlp:
                    # For MLP, we only need features of output nodes
                    batch_pred = model(features[output_nodes])
                else:
                    # For graph models, we use the subgraph (blocks)
                    batch_pred = model(blocks, batch_inputs)

                # Compute loss
                loss = F.cross_entropy(batch_pred, batch_labels)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

        else:
            for batch_graph, batch_nodes in batchify(graph, train_idx, batch_size):
                batch_count += 1

                # Get the features and labels for this batch
                batch_inputs = features[batch_nodes]
                batch_labels = labels[batch_nodes].float()
                batch_inputs = batch_inputs.to(device)
                batch_labels = batch_labels.to(device)

                # Forward pass - different handling for MLP and graph models
                if is_mlp:
                    # For MLP, we only need features of output nodes
                    batch_pred = model(batch_inputs)
                else:
                    # For graph models, we use the subgraph (blocks)
                    batch_pred = model(batch_graph, batch_inputs)

                # Compute loss
                loss = F.cross_entropy(batch_pred, batch_labels)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

        # After each epoch, evaluate the model on the validation set - only on val_idx
        model.eval()
        train_acc, train_f1 = evaluate(model, graph, features, labels, train_idx, is_mlp=is_mlp)
        val_acc, val_f1 = evaluate(model, graph, features, labels, val_idx, is_mlp=is_mlp)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/batch_count:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
            best_f1 = val_f1
            best_val_epoch = epoch +1
            print(f"Best model updated at epoch {epoch+1} with val acc: {best_val_acc:.4f}, val f1: {best_f1:.4f}")
        if train_acc > best_train_acc:
            best_train_acc = train_acc
            best_train_f1 = train_f1


    return best_model_state, train_acc, train_f1, best_val_acc, best_f1, best_val_epoch

def batchify(graph, train_idx, batch_size):
    #if not on device
    graph = graph.to(device)
    train_idx = train_idx.to(device)

    for i in range(0,len(train_idx),batch_size):
        batch_nodes = train_idx[i:i+batch_size]
        subgraph = dgl.node_subgraph(graph, batch_nodes)
        yield subgraph, batch_nodes
