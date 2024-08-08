# GraphAlign: Pretraining One Graph Neural Network on Multiple Graphs via Feature Alignment

Paper link: [arxiv](https://arxiv.org/abs/2406.02953)

## Dependencies

1.  PyTorch >= v2.1.1 and CUDA >= 11.4. Older and newer pytorch versions are both competable.

2. [dgl](https://www.dgl.ai/pages/start.html) >= 0.7.2

3. [localclustering](https://github.com/kfoynt/LocalGraphClustering) (optional for data preprocessing)

4. Run `./setup.sh` in Treminal to install other dependencies.

## Dataset Preprocessing

For Large scale graphs, before starting mini-batch training, you'll need to generate local clusters if you want to use local-clustering for training. To generate a local cluster, you should first install [localclustering](https://github.com/kfoynt/LocalGraphClustering) and then run the following command:

 ````python
 python generate_data.py \
 	--data_save_path <path/to/data_dir> \
 	--device <gpu_id> \
 	--batch_size 512 \
 	--dataset_name ogbn-arxiv ogbn-products ogbn-papers100M FB15K237 Cora WN18RR
 ````

And we also provide the pre-generated local clusters which can be downloaded [here](https://cloud.tsinghua.edu.cn/d/ab2d7eb8ff524554b926/) (coming soon) for usage.

## Quick Start

To reproduce individually pretraining results, run: (first param is device, second param is path/to/save/data)
```bash
bash scripts/individually_pretrain.sh 0 your/path/to/save/data
```

To reproduce GraphAlign results, run: (first param is device, e.g. "0,1" means using device 0 and 1, second param is path/to/save/data)
```bash
# For GNN pretraining
bash scripts/graphalign.sh <gpu_ids> <path/to/data>

# Evaluation after GNN pretraining checkpoint
bash scripts/evaluation.sh <gpu_id> <path/to/data> <path/to/gnn_ckpt>
```

To reproduce few-shot results:
```bash
# Evaluate the pretraing GNN in few-shot classification
bash scripts/few_shot_eval.sh <gpu_id> <path/to/data> </path/to/gnn_ckpt>
```

## Experimental Results

- Linear probing results in unsupervised representation learning for node classification

| Method    | Setting                  | ogbn-arxiv     | ogbn-products  | ogbn-papers100M | Avg. gain |
| --------- | ------------------------ | -------------- | -------------- | --------------- | --------- |
| MLP       | supervised               | 69.85±0.36     | 73.74±0.43     | 56.62±0.21      | -         |
| GAT       | supervised               | 74.15±0.15     | 83.42±0.35     | 66.63±0.23      | -         |
| GCN       | supervised               | 74.77±0.34     | 80.76±0.50     | 68.15±0.08      | -         |
| SGC       | supervised               | 71.56±0.41     | 74.36±0.27     | 58.82±0.08      | -         |
| BGRL      | individually-pretrain    | 72.98±0.14     | 80.45±0.16     | 65.40±0.23      | -         |
|           | vanilla jointly-pretrain | 69.00±0.08     | 81.11±0.27     | 63.93±0.22      | -1.60     |
|           | **GraphAlign**           | **73.20±0.20** | **80.79±0.45** | **65.62±0.14**  | **+0.26** |
| GRACE     | individually-pretrain    | 73.33±0.19     | 81.91±0.27     | 65.59±0.13      | -         |
|           | vanilla jointly-pretrain | 72.10±0.18     | 81.96±0.34     | 65.54±0.18      | -0.41     |
|           | **GraphAlign**           | **73.69±0.26** | **81.90±0.19** | **65.61±0.17**  | **+0.12** |
| GraphMAE  | individually-pretrain    | 72.35±0.12     | 81.69±0.11     | 65.68±0.28      | -         |
|           | vanilla jointly-pretrain | 71.98±0.24     | 82.36±0.19     | 65.92±0.13      | +0.18     |
|           | **GraphAlign**           | **72.97±0.22** | **82.51±0.18** | **66.08±0.18**  | **+0.61** |
| GraphMAE2 | individually-pretrain    | 73.10±0.11     | 82.53±0.17     | 66.28±0.10      | -         |
|           | vanilla jointly-pretrain | 71.28±0.25     | 80.05±0.35     | 64.28±0.33      | -2.10     |
|           | **GraphAlign**           | **73.56±0.26** | **82.93±0.42** | **66.39±0.14**  | **+0.32** |

- Few-shot node classification

  ## Few-shot node classification results on ogbn-arxiv and Cora, and link classification results on FB15K237 and WN18RR

  We report $m$-way-$k$-shot accuracy(%), 5-way for ogbn-arxiv, Cora, WN18RR and 20-way for FB15K237.

  | Method                | ogbn-arxiv 5-shot | ogbn-arxiv 1-shot | Cora 5-shot | Cora 1-shot | WN18RR 5-shot | WN18RR 1-shot | FB15K237 5-shot | FB15K237 1-shot |
  | --------------------- | ----------------- | ----------------- | ----------- | ----------- | ------------- | ------------- | --------------- | --------------- |
  | GPN                   | 50.53±3.07        | 38.58±1.61        | -           | -           | -             | -             | -               | -               |
  | TENT                  | 60.83±7.45        | 45.62±10.70       | -           | -           | -             | -             | -               | -               |
  | GLITTER               | 56.00±4.40        | 47.12±2.73        | -           | -           | -             | -             | -               | -               |
  | Prodigy               | 61.09±5.85        | 48.23±6.18        | -           | -           | -             | -             | 74.92±6.03      | 55.49±6.88      |
  | OFA                   | 61.45±2.56        | 50.20±4.27        | 48.76±2.65  | 34.04±4.10  | 46.32±4.18    | 33.86±3.41    | 82.56±1.58      | 75.39±2.86      |
  | _OFA-emb-only        | 61.27±7.09        | 43.22±8.45        | 58.60±6.72  | 40.87±8.26  | 54.87±9.73    | 39.72±9.35    | 59.11±6.95      | 43.03±7.17      |
  |     | | | | | ||||
  | **GraphAlign**(GraphMAE) | 81.93±6.22        | 65.02±10.62       | 74.49±6.43  | 55.55±9.86  | 60.19±10.31   | 45.08±10.55   | 79.92±5.54      | 63.01±7.29      |
  | **GraphAlign**(GraphMAE2) | 83.97±5.85        | 70.65±10.45       | 73.66±6.75  | 56.87±9.98  | 55.95±10.49   | 42.22±10.04   | 79.86±5.53      | 63.56±7.31      |
  | **GraphAlign**(GRACE) | 84.76±5.71        | 71.18±10.29       | 69.85±7.19  | 52.60±10.10 | 53.11±10.24   | 39.58±9.42    | 75.04±5.98      | 60.09±7.36      |
  | **GraphAlign**(BGRL)  | 81.88±6.26        | 66.31±10.63       | 68.13±6.84  | 50.19±9.49  | 51.97±10.66   | 38.72±9.77    | 77.74±5.87      | 61.48±7.44      |
  | E5-emb-only           | 65.67±7.02        | 47.13±8.68        | 59.71±6.71  | 41.58±8.11  | 56.52±9.65    | 41.53±9.36    | 58.43±6.94      | 42.06±7.11      |

## Citing

```latex
@article{hou2024graphalign,
  title={GraphAlign: Pretraining One Graph Neural Network on Multiple Graphs via Feature Alignment},
  author={Hou, Zhenyu and Li, Haozhan and Cen, Yukuo and Tang, Jie and Dong, Yuxiao},
  journal={arXiv preprint arXiv:2406.02953},
  year={2024}
}
```

