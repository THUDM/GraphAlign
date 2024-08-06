# GraphAlign

## Installing

### Dependencies

1. Install PyTorch with CUDA. The repository is tested with PyTorch
v2.1.1 and CUDA 11.4. Older and newer pytorch versions are both competable.

2. Install [dgl](https://www.dgl.ai/pages/start.html).

3. (Optional) For Ogb-dataset, mini-batch training requires localcluster_ego_subgraphs. 
If you want to generate localcluster_ego_subgraphs by [generate_data.py](generate_data.py), you need to install [localclustering](https://github.com/kfoynt/LocalGraphClustering) first.

4. Run `./setup.sh` in Treminal to install other dependencies.

### Prepare Dataset

5. Run `python generate_data.py --data_save_path your/path/to/save/data --device cuda:cuda_number --batch_size 512 --dataset_name ogbn-arxiv ogbn-products ogbn-papers100M FB15K237 Cora WN18RR`. Processing time for ogbn-papers100M will be 24-36 hours depending on gpu and cpu. We recommend to download our prepared data.

6. Or download prepared data from thislink and put the data under `your/path/to/save/data`.

## Run

To reproduce individually pretraining results, run: (first param is device, second param is path/to/save/data)
```bash
./scripts/individually_pretrain.sh 0 your/path/to/save/data
```

To reproduce GraphAlign results, run: (first param is device, e.g. "0,1" means using device 0 and 1, second param is path/to/save/data)
```bash
./scripts/graphalign.sh 0,1 your/path/to/save/data
(After get the pretrained GNN checkpoint, run:)
./scripts/evaluation.sh 0 your/path/to/save/data your/path/to/save/gnn/checkpoint
```

To reproduce few-shot results, run:(first param is device, second param is path/to/save/data, third param is path/to/save/gnn/checkpoint)
```bash
./scripts/few_shot_eval.sh 0 your/path/to/save/data your/path/to/save/gnn/checkpoint
```
