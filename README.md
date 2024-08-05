# GraphAlign

### Prerequisites

PyTorch with CUDA is required. The repository is tested with PyTorch
v2.1.1 and CUDA 11.4. Older and newer pytorch versions are both competable.

The minimum version of supported PyTorch is `1.7.2` with CUDA `10`. However,
there are a few known issues that requires manual modification of FastMoE's
code with specific older dependents.

If the distributed expert feature is enabled, NCCL with P2P communication
support, typically versions `>=2.7.5`, is needed. 

### Installing

## Preparation

1. Download the processed datasets directly or download the origin datasets and process them. 

2. Install Python dependencies.

## Run

To reproduce individually pretraining results, run:
```bash
./scripts/individually_pretrain.sh
```

To reproduce GraphAlign results, run:
```bash
./scripts/graphalign.sh
(After get the pretrained GNN checkpoint, run:)
./scripts/evaluation.sh
```

To reproduce few-shot results, run:
```bash
./scripts/few_shot_eval.sh
```
