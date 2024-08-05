import torch
import torch.nn as nn
from torch import optim as optim
import dgl
import random
import numpy as np
import argparse
import yaml
import psutil
import os
import sys
import wandb
from sklearn.metrics import f1_score,  recall_score, precision_score


def create_optimizer(opt, model, lr, weight_decay):
    opt_lower = opt.lower()
    parameters = model.parameters()
    opt_args = dict(lr=lr, weight_decay=weight_decay)
    opt_split = opt_lower.split("_")
    opt_lower = opt_split[-1]

    if opt_lower == "adam":
        return optim.Adam(parameters, **opt_args)
    elif opt_lower == "adamw":
        return optim.AdamW(parameters, **opt_args)
    elif opt_lower == "adadelta":
        return optim.Adadelta(parameters, **opt_args)
    elif opt_lower == "radam":
        return optim.RAdam(parameters, **opt_args)
    elif opt_lower == "sgd":
        opt_args["momentum"] = 0.9
        return optim.SGD(parameters, **opt_args)
    else:
        assert False and "Invalid optimizer"


def create_scheduler(args, optimizer):
    if not args.no_verbose:
        print("Use schedular")
    if args.model == "bgrl":
        scheduler = lambda epoch: epoch / 1000 if epoch < 1000 \
            else (1 + np.cos((epoch - 1000) * np.pi / (args.max_epoch - 1000))) * 0.5
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
    else:
        scheduler = lambda epoch: (1 + np.cos((epoch) * np.pi / args.max_epoch)) * 0.5
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)


def create_optimizer_learn_feature(opt, model, lr, weight_decay, in_feature):
    opt_lower = opt.lower()
    parameters = [{'params': model.parameters()}, {'params': in_feature}]
    opt_args = dict(lr=lr, weight_decay=weight_decay)
    opt_split = opt_lower.split("_")
    opt_lower = opt_split[-1]

    if opt_lower == "adam":
        return optim.Adam(parameters, **opt_args)
    elif opt_lower == "adamw":
        return optim.AdamW(parameters, **opt_args)
    elif opt_lower == "adadelta":
        return optim.Adadelta(parameters, **opt_args)
    elif opt_lower == "radam":
        return optim.RAdam(parameters, **opt_args)
    elif opt_lower == "sgd":
        opt_args["momentum"] = 0.9
        return optim.SGD(parameters, **opt_args)
    else:
        assert False and "Invalid optimizer"


def create_optimizer_roberta(opt, model, lr, weight_decay):
    opt_lower = opt.lower()
    if hasattr(model, 'encoder'):
        parameters = [{'params': model.encoder.parameters(), lr: lr}, {'params': model.classifier.parameters(), lr: lr},
                      {'params': model.roberta.parameters(), lr: 1e-7}]
    else:
        parameters = [{'params': model.classifier.parameters(), lr: lr},
                      {'params': model.roberta.parameters(), lr: 1e-7}]
    opt_args = dict(lr=lr, weight_decay=weight_decay)
    opt_split = opt_lower.split("_")
    opt_lower = opt_split[-1]

    if opt_lower == "adam":
        return optim.Adam(parameters, **opt_args)
    elif opt_lower == "adamw":
        return optim.AdamW(parameters, **opt_args)
    elif opt_lower == "adadelta":
        return optim.Adadelta(parameters, **opt_args)
    elif opt_lower == "radam":
        return optim.RAdam(parameters, **opt_args)
    elif opt_lower == "sgd":
        opt_args["momentum"] = 0.9
        return optim.SGD(parameters, **opt_args)
    else:
        assert False and "Invalid optimizer"


def get_current_lr(optimizer):
    return optimizer.state_dict()["param_groups"][0]["lr"]


def process_args(args):    
    if args.use_cfg:
        path = args.use_cfg_path
        if path == "":
            if args.model == "graphmae":
                path = "./configs/GraphMAE_configs.yml"
            elif args.model == "graphmae2":
                path = "./configs/GraphMAE2_configs.yml"
            elif args.model == "grace":
                path = "./configs/Grace_configs.yml"
            elif args.model == "cca_ssg":
                path = "./configs/CCA_SSG_configs.yml"
            elif args.model == "bgrl":
                path = "./configs/BGRL_configs.yml"
            elif args.model == "ggd":
                path = "./configs/GGD_configs.yml"
        with open(path, "r") as f:
            configs = yaml.load(f, yaml.FullLoader)
        if args.dataset not in configs:
            return args
        configs = configs[args.dataset]
        for k, v in configs.items():
            if "lr" in k or "weight_decay" in k or "tau" in k or "lambd" in k:
                v = float(v)
            setattr(args, k, v)
    
    if not args.logging_path.endswith('.log'):
        os.makedirs(args.logging_path, exist_ok=True)
        args.logging_path = os.path.join(args.logging_path,
                                         f"{args.model}_{args.dataset}_lc.log")   
    return args


class Logger(object):
    def __init__(self, no_verbose=False):
        self.terminal = sys.stdout
        self.log = None
        self.no_verbose = no_verbose

    def set_log_path(self, filename="Default.log"):
        print(f"Writing logs to {filename}")
        if self.log is not None:
            self.log.close()
        self.log = open(filename, "a")

    def write(self, message):
        if not self.no_verbose:
            self.terminal.write(message)
        self.log.write(message)
        sys.stdout.flush()

    def flush(self):
        pass


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.determinstic = True
    dgl.random.seed(seed)


# training on down stream tasks
class LogisticRegression(nn.Module):
    def __init__(self, num_dim, num_class):
        super().__init__()
        self.linear = nn.Linear(num_dim, num_class)

    def forward(self, x):
        logits = self.linear(x)
        return logits
 
def accuracy(y_pred, y_true):
    y_true = y_true.squeeze().long()
    preds = y_pred.max(1)[1].type_as(y_true)
    correct = preds.eq(y_true).double()
    correct = correct.sum().item()
    return correct / len(y_true)

def F1_score(y_pred, y_true):
    y_true = y_true.squeeze().long()
    y_pred = y_pred.max(1)[1].type_as(y_true)
    y_pred = y_pred.cpu().numpy().astype('int32')
    y_true = y_true.cpu().numpy().astype('int32')
    
    f1 = f1_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    return f1, recall, precision

def show_occupied_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 ** 2


# build args for all models
def build_args():
    parser = argparse.ArgumentParser(description="SSL-GNN-Train Settings")
    parser.add_argument("--model", type=str, default="graphmae")
    parser.add_argument("--linear_prob_seeds", type=int, nargs="+", default=[i for i in range(10)],
                        help="seed for linear_prob and few-shot")
    parser.add_argument("--pretrain_seeds", type=int, nargs="+", default=[i for i in range(1)],
                        help="pretrian seeds")
    parser.add_argument("--dataset", type=str, default="ogbn-products")
    parser.add_argument("--device", type=int, default=0)

    # just for passing arguments, no need to set/change
    parser.add_argument("--num_features", type=int, default=-1)
    parser.add_argument("--logging_path", type=str, default='logs', help='path to save .log')
    # encoder parameters
    parser.add_argument("--num_heads", type=int, default=4, help="number of hidden attention heads")
    parser.add_argument("--num_layers", type=int, default=2, help="number of hidden layers")
    parser.add_argument("--num_hidden", type=int, default=256, help="number of hidden units")
    parser.add_argument("--residual", action="store_true", default=False, help="use residual connection")
    parser.add_argument("--in_drop", type=float, default=.2, help="input feature dropout")
    parser.add_argument("--attn_drop", type=float, default=.1, help="attention dropout")
    parser.add_argument("--norm", type=str, default=None)
    parser.add_argument("--negative_slope", type=float, default=0.2, help="the negative slope of leaky relu for GAT")
    parser.add_argument("--activation", type=str, default="prelu")

    # parameters of GraphMAE
    parser.add_argument("--mask_rate", type=float, default=0.5)
    parser.add_argument("--drop_edge_rate", type=float, default=0.0)
    parser.add_argument("--replace_rate", type=float, default=0.0)
    parser.add_argument("--encoder", type=str, default="gat")
    parser.add_argument("--decoder", type=str, default="gat")
    parser.add_argument("--loss_fn", type=str, default="sce")
    parser.add_argument("--alpha_l", type=float, default=2, help="`pow`coefficient for `sce` loss")
    parser.add_argument("--concat_hidden", action="store_true", default=False)
    parser.add_argument("--num_out_heads", type=int, default=1, help="number of output attention heads")
    # rest parameters in GraphMAE2
    parser.add_argument("--num_dec_layers", type=int, default=1)
    parser.add_argument("--num_remasking", type=int, default=3)
    parser.add_argument("--remask_rate", type=float, default=0.5)
    parser.add_argument("--remask_method", type=str, default="random")
    parser.add_argument("--mask_method", type=str, default="random")
    parser.add_argument("--mask_type", type=str, default="mask", help="`mask` or `drop`")
    parser.add_argument("--drop_edge_rate_f", type=float, default=0.0)
    parser.add_argument("--label_rate", type=float, default=1.0)
    parser.add_argument("--lam", type=float, default=1.0)
    parser.add_argument("--delayed_ema_epoch", type=int, default=0)
    parser.add_argument("--momentum", type=float, default=0.996)

    # parameters of Grace
    parser.add_argument("--num_proj_hidden", type=int, default=1, help="h->z linear in grace")
    parser.add_argument("--drop_edge_rate_1", type=float, default=0.5)
    parser.add_argument("--drop_edge_rate_2", type=float, default=0.5)
    parser.add_argument("--drop_feature_rate_1", type=float, default=0.5)
    parser.add_argument("--drop_feature_rate_2", type=float, default=0.5)
    parser.add_argument("--tau", type=float, default=0.5)

    # parameters of CCA-SSG
    parser.add_argument('--der', type=float, default=0.2, help='Drop edge ratio.')
    parser.add_argument('--dfr', type=float, default=0.2, help='Drop feature ratio.')
    parser.add_argument('--lambd', type=float, default=1e-3, help='trade-off ratio.')

    # parameters of BGRL (4 drop rate use Grace's arguments)
    parser.add_argument('--pred_hid', type=int, default=256)
    parser.add_argument("--moving_average_decay", type=float, default=0.99)
    parser.add_argument('--ema_total_steps', type=int, default=1000)
    
    # parameters of ggd
    parser.add_argument('--drop_feat', type=float, default=0.1)
    parser.add_argument("--num_proj_layers", type=int, default=1, help="number of project linear layers")

    # pretraining settings
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--scheduler", action="store_true", default=False)
    parser.add_argument("--warmup_steps", type=int, default=-1)
    parser.add_argument("--max_epoch", type=int, default=200, help="number of training epochs")
    parser.add_argument("--lr", type=float, default=0.005, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="weight decay")

    # eval settings
    parser.add_argument("--linear_prob", action="store_true", default=False)
    parser.add_argument("--max_epoch_f", type=int, default=30)
    parser.add_argument("--lr_f", type=float, default=0.001, help="learning rate for evaluation")
    parser.add_argument("--weight_decay_f", type=float, default=0.0, help="weight decay for evaluation")

    # graph batch training settings
    parser.add_argument("--batch_size", type=int, default=512, help="batch_size in pretraining")
    parser.add_argument("--batch_size_f", type=int, default=256, help="batch_size_f is batch_size in inference")
    parser.add_argument("--batch_size_linear_prob", type=int, default=5120, help="batch_size_linear_prob is batch_size to train linear in linear_prob")
    # local clustering
    parser.add_argument("--ego_graph_file_path", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default="data")
    # other settings
    parser.add_argument("--eval_first", action="store_true")
    parser.add_argument("--load_model", action="store_true")
    parser.add_argument("--load_model_path", type=str, default="")
    parser.add_argument("--save_model", action="store_false")
    parser.add_argument("--save_model_path", type=str, default="GraphAlign-checkpoints")
    parser.add_argument("--use_cfg", action="store_true")
    parser.add_argument("--use_cfg_path", type=str, default="")
    parser.add_argument("--no_verbose", action="store_true", help="do not print process info")
    parser.add_argument("--no_scale", action="store_true")
    parser.add_argument("--multi_scale", action="store_true")
    parser.add_argument("--feat_type", type=str, default="e5_float16") #linkpredict
    parser.add_argument("--weight", type=int, nargs="+", default=None)
    parser.add_argument("--default_dataset", type=str, default=None) 
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("--num_expert", type=int, default=4)
    parser.add_argument("--hiddenhidden_size_times", type=int, default=1)
    parser.add_argument("--deepspeed", action="store_true")
    parser.add_argument("--deepspeed_config", type=str, default=None)
    parser.add_argument("--pretrain_dataset", type=str, nargs="+", default=None)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--pretrain_num_workers", type=int, default=4)
    parser.add_argument("--moe", action="store_true")
    parser.add_argument("--prob_num_workers", type=int, default=4)
    parser.add_argument("--moe_use_linear", action="store_true")

    parser.add_argument("--decoder_no_moe", action="store_true")
    parser.add_argument("--moe_layer", type=int, nargs="+", default=None)
    parser.add_argument("--dataset_drop_edge", type=str, nargs="+", default=["ogbn-arxiv","ogbn-products","ogbn-papers100M"])
    parser.add_argument("--log_each_dataset_loss", action="store_true")
    parser.add_argument("--drop_model", type=str, default="random")
    # for graphmae2
    parser.add_argument("--graphmae2_ema_graph_nodrop", action="store_true")
    #for grace and bgrl
    parser.add_argument("--dataset_drop_feat", type=str, nargs="+", default=None)
    #few shot 
    parser.add_argument("--few_shot", action="store_true")
    parser.add_argument("--eval_num_label", type=int, default=5)
    parser.add_argument("--eval_num_support", type=int, default=5)
    parser.add_argument("--eval_num_query", type=int, default=20)
    parser.add_argument("--khop", type=int, default=1)
    parser.add_argument("--total_steps", type=int, default=500)
    parser.add_argument("--sample_position", type=str, default="total") #train_test
    parser.add_argument("--fs_label", type=str, default="total") #ofa    
    args = parser.parse_args()
    return args


def print_rank_0(message):
    """If distributed is initialized, print only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)
