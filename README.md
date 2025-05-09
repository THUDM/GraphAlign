# Exploration into cross-domain task generalization of graphs with language embeddings

## ABSTRACT
Scaling and generalizing cross-domain tasks amongst graph datasets remains a challenge due to the variability in node features, edge-based relationships, and the inherit challenges for transfer-learning amongst graphs. The aim of this project is to explore the capabilities of using language embeddings to achieve task generalizations across different graph structures and build models that could learn cross-domain relationships. By evaluating the performance of joinly trained graph neural networks across different language embeddings, the authors evaluate the effectiveness of various encoding architectures. Contrary to expectations, the simpler word2vec achieved greater performance compared to the E5-Small-V2 and GraphAlign pretrained embeddings. Finally, the author discusses limitations and the conclusiveness of the study and discusses future research directions in unifying cross-domain graphs with scalable architecture.


## OVERVIEW
This project provides tools for:
- Generating text embeddings for graph nodes using various language models
- Training Graph Neural Networks on these embeddings
- Evaluating model performance on node classification tasks
- The implementation supports multiple datasets (ogbn-arxiv, ogbn-mag, combined) and embedding methods (E5, GraphAlign).

### Requirements
- Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Train GNNs using preprocessed datasets

This module currently supports multiple graph datasets, which can be passed as an argument --graph_idx. Two graph sampling methods are used during training process a node-batching ("batchify") function and a multi-hop sample. This can be toggled with --sample as well.

```python
GRAPHS = ['combined_graph_pca.bin','graph0.bin','graph1.bin','pca_graph.bin']
```

```bash
# Train using batch sampling
python graph_train.py --graph_idx 0

# Train using multi-hop neighbor sampling
python graph_train.py --graph_idx 0 --sample

# Train using multi-hop neighbor sampling
python graph_train.py --graph_idx 0 --sample
```


## DATASET INFO

<img width="589" alt="image" src="https://github.com/user-attachments/assets/b7926e35-7417-4b50-b45d-7c47cb92bc8e" />
<img width="584" alt="image" src="https://github.com/user-attachments/assets/98d8d5ea-8a00-409b-98f8-230c970fb8f1" />

### baseline of original embedding in paper obg:

<img width="418" alt="image" src="https://github.com/user-attachments/assets/7f4ddee9-1fc7-4052-ad7c-e54f0ded301f" />
<img width="385" alt="image" src="https://github.com/user-attachments/assets/4f12edb2-fecb-473e-a800-4174ebb20b9b" />

<img width="442" alt="image" src="https://github.com/user-attachments/assets/82e65470-45da-4cf4-9b42-0e92e8c05b37" />
