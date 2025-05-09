import trainer_large
from src.utils.utils import build_args, process_args,Logger
import sys
import json
import wandb
import numpy as np
import deepspeed
import torch
import time

def train_eval(args):
    if args.dataset.startswith("ogbn") or args.dataset in ["Wiki","ConceptNet","FB15K237","Cora","WN18RR"] or args.dataset=="graphalign" :
        trainer = trainer_large.ModelTrainer(args)
        acc_list = []
        for pretrain_seed in args.pretrain_seeds:
            acc = trainer.train_eval(pretrain_seed=pretrain_seed)
            acc_list = acc_list+acc
        final_test_acc, final_test_acc_std = np.mean(acc_list), np.std(acc_list)
        print(f"Fin Average final-test-acc: {final_test_acc:.4f}±{final_test_acc_std:.4f}", end="")
        wandb.summary[f'Fin Average final-test-acc'] = final_test_acc
        wandb.summary[f'Fin Average final-test-acc-std'] = final_test_acc_std

    else:
        raise NotImplementedError(f"{args.dataset} is not supported yet!")

def main():
    args = build_args()
    args = process_args(args)
    train_eval(args)
    wandb.finish()

if __name__ == "__main__":
    main()




