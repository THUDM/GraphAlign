import copy
import numpy as np
from tqdm import tqdm
import torch
import torch.distributed as dist
import os
# models
from models import build_model

# self-made utils
from utils.load_data import load_dataloader, LinearProbingDataLoader, setup_incontext_dataloader
from utils.utils import (
    create_optimizer,
    set_random_seed,
    create_scheduler,
    LogisticRegression,
    accuracy,
    F1_score,
    show_occupied_memory,
    print_rank_0
)
import warnings
import wandb
import deepspeed
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
import time


warnings.filterwarnings("ignore")

def wconfig_2_args(config, args):
    for k,v in config.items():
        setattr(args, k, v)
    return args


def get_save_path_name(args,pretrain_seed):
    short_title={"ogbn-arxiv":"arx", "ogbn-products":"prod", "ogbn-papers100M":"100M", "random":"rand", "directed_to_undirected":"dir2undir", "ran_and_di2undi":"ran_and_di2undi", "Wiki":"Wiki", "ConceptNet":"concept", "FB15K237":"FB15",
                  "graphmae":"mae","graphmae2":"mae2","bgrl": "bgrl","grace":"grace" }
    model_name =""
    if args.moe:
        model_name = model_name + f"{short_title[args.model]}_use_moe"
    else:
        model_name = model_name + f"{short_title[args.model]}_no_moe"

    if args.moe_use_linear:
        assert args.moe==True
        model_name = model_name+"_linear"
    
    if args.moe:
        if args.moe_layer==None:
            moe_layer_name = "_lall"
        else:
            moe_layer_name = "_l"
            for l in args.moe_layer:
                moe_layer_name = moe_layer_name+f"{l}"
        model_name = model_name+moe_layer_name
    
    for x in args.pretrain_dataset:
        model_name = model_name + "_" + short_title[x]
        
    if args.weight == None:
        w_name = "None_"
    else:
        w_name = ""
        for w in args.weight:
            w_name = w_name + f"{w}_"
    
    if args.model in ["graphmae","graphmae2"]:
        model_name = model_name + f"_weights_{w_name}pseed_{pretrain_seed}_topk_{args.top_k}_numexp_{args.num_expert}_hstime_{args.hiddenhidden_size_times}_mask_{args.mask_rate}_alpha_{args.alpha_l}_GNNlayers_{args.num_layers}_lr_{args.lr}_decay_{args.weight_decay}_epoch_{args.max_epoch}_edgedrop_{args.drop_edge_rate}_method_{short_title[args.drop_model]}_"
    elif args.model in ["grace"]:
        model_name = model_name + f"_weights_{w_name}pseed_{pretrain_seed}_topk_{args.top_k}_numexp_{args.num_expert}_hstime_{args.hiddenhidden_size_times}_GNNlayers_{args.num_layers}_lr_{args.lr}_decay_{args.weight_decay}_epoch_{args.max_epoch}_tau_{args.tau}_featdrop_{args.drop_feature_rate_1}_{args.drop_feature_rate_2}_edgedrop_{args.drop_edge_rate}_method_{short_title[args.drop_model]}_"
    elif args.model in ["bgrl"]:
        model_name = model_name + f"_weights_{w_name}pseed_{pretrain_seed}_topk_{args.top_k}_numexp_{args.num_expert}_hstime_{args.hiddenhidden_size_times}_GNNlayers_{args.num_layers}_lr_{args.lr}_decay_{args.weight_decay}_epoch_{args.max_epoch}_featdrop_{args.drop_feature_rate_1}_{args.drop_feature_rate_2}_edgedrop_{args.drop_edge_rate}_method_{short_title[args.drop_model]}_"
    elif args.model in ["ggd"]:
        model_name = model_name + f"_weights_{w_name}pseed_{pretrain_seed}_topk_{args.top_k}_numexp_{args.num_expert}_hstime_{args.hiddenhidden_size_times}_GNNlayers_{args.num_layers}_lr_{args.lr}_decay_{args.weight_decay}_epoch_{args.max_epoch}_featdrop_{args.drop_feat}_edgedrop_{args.drop_edge_rate}_method_{short_title[args.drop_model]}_"
    else:
        raise ValueError

    for x in args.dataset_drop_edge:
        model_name = model_name  + short_title[x] + "_"

    if args.graphmae2_ema_graph_nodrop:
        model_name = model_name + "ema_nodrop_"

    if args.decoder_no_moe:
        model_name = model_name+ "dec_nomoe_"

    if args.no_scale:
        model_name = model_name+ "no_scale_"
    else:
        model_name = model_name+ "scale_" 

    if args.default_dataset!=None:
        model_name = model_name + f"samplebase_{args.default_dataset}_"    
    
    if args.feat_type in ["e5_float16","e5_float32"]:
        model_name = model_name + f"e5_" 
    elif args.feat_type in ["ofa_float16","ofa_float32"]:  
        model_name = model_name + f"ofa_" 
    else:
        raise ValueError
    
    model_name = model_name + args.encoder + "_"

    model_name = model_name + "checkpoint"        
    return model_name

def data_loading_thread(data_queue, dataloader):
    for batch in dataloader:
        data_queue.put(batch)

class ModelTrainer:
    def __init__(self, args):
        self._args = args
        self._device = args.device if args.device >= 0 else "cpu"
    
    def train_eval(self,pretrain_seed=0):
        args = self._args
        set_random_seed(pretrain_seed)
        if args.deepspeed:
            deepspeed.init_distributed()

        print_rank_0(args)
        #
        if args.moe: 
            project_name = f"{args.model}-mixpretrain+moe"
        elif args.few_shot:
            project_name = f"{args.model}-fewshot-{args.dataset}"
        else:
            project_name = f"{args.model}-mixpretrain"
        #
        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:  
                wandb.init(project=f"SSL-{args.dataset}",name=project_name ,config=vars(args),dir="../")
        else:
            wandb.init(project=f"SSL-{args.dataset}",name=project_name ,config=vars(args),dir="../")      
        self._args = args  

        memory_before_load = show_occupied_memory()
        if not args.load_model :
            self._pretrain_dataloader = load_dataloader("pretrain", args.dataset, self._args, pretrain_seed=pretrain_seed)
            self._args.ema_total_steps = len(self._pretrain_dataloader)*self._args.max_epoch
            args.ema_total_steps = len(self._pretrain_dataloader)*self._args.max_epoch
            
        if args.feat_type in ["e5_float16","e5_float32"]:
            self._args.num_features = 384
        elif args.feat_type in ["ofa_float16","ofa_float32"]: 
            self._args.num_features = 768
        elif args.feat_type in ["origin_float16","origin_float32"]: 
            self._args.num_features = 128
        else:
            raise ValueError
        print_rank_0(f"Data memory usage: {show_occupied_memory() - memory_before_load:.2f} MB")
        
        self.model = build_model(self._args)
        self.optimizer = create_optimizer(args.optimizer, self.model, args.lr, args.weight_decay)
        self.scheduler = None
        if args.scheduler:
            self.scheduler = create_scheduler(args, self.optimizer)
        if args.deepspeed:
            self.model, self.optimizer, _, _ = deepspeed.initialize(
                        args=args,
                        model=self.model, 
                        optimizer=self.optimizer, 
                        lr_scheduler=self.scheduler
                    )
        else:
            self.model.to(args.device)
        self._device = next(self.model.parameters()).device

        # need to pretrain
        if not args.load_model:
            if args.deepspeed:
                ckpt_dir = os.path.join(args.data_dir ,args.save_model_path, "ds_" + get_save_path_name(args,pretrain_seed))
            else:
                ckpt_dir = os.path.join(args.data_dir ,args.save_model_path)   
            os.makedirs(args.data_dir, exist_ok=True)
            os.makedirs(os.path.join(args.data_dir ,args.save_model_path), exist_ok=True)
            os.makedirs(ckpt_dir, exist_ok=True)
           
            self.pretrain(ckpt_dir)

            if args.save_model:
                if args.deepspeed:
                    if dist.get_rank() == 0:
                        save_path = os.path.join(args.data_dir ,args.save_model_path, get_save_path_name(args,pretrain_seed) + ".pt")
                        model = self.model.module.cpu()
                        print(f"Saving model to {save_path}")
                        torch.save(model.state_dict(), save_path)
                else:
                    model = self.model.cpu()
                    save_path = os.path.join(ckpt_dir, get_save_path_name(args,pretrain_seed) + ".pt")
                    print(f"Saveing model to {save_path}")
                    torch.save(model.state_dict(), save_path)
        else:
            print(f"Loading model from {args.load_model_path}")
            self.model.load_state_dict(torch.load(args.load_model_path))
        
        if args.deepspeed:
            print("pretrain finished")
            exit(0)

        print("---- start evaluation ----")
        acc_list = []
        set_random_seed(0)

        if args.few_shot:
            print("---- start few-shot ----")
            acc = self.incontext_evaluate()
        else:
            self.infer_embeddings()
            acc = self.evaluate()
        acc_list = acc_list + acc
        final_test_acc, final_test_acc_std = np.mean(acc_list), np.std(acc_list)
        print(f"#pretrain seed{pretrain_seed} final-test-acc: {final_test_acc:.4f}±{final_test_acc_std:.4f}", end="")
        wandb.summary[f'pretrain seed{pretrain_seed} final-test-acc'] = final_test_acc
        wandb.summary[f'pretrain seed{pretrain_seed} final-test-acc-std'] = final_test_acc_std
        return acc_list

    def pretrain(self, ckpt_dir):
        args = self._args
        print_rank_0(f"\n--- Start pretraining {args.model} model on {args.dataset} using lc sampling ---")
        client_sd = {}
        step = 0
        data_queue = queue.Queue(maxsize=15)
        for epoch in range(args.max_epoch):
            self.model.train()
            queue_size = data_queue.qsize()
            assert queue_size == 0
            data_thread = threading.Thread(target=data_loading_thread, args=(data_queue,self._pretrain_dataloader,))
            data_thread.start()
            epoch_iter = tqdm(range(len(self._pretrain_dataloader)))
            losses = []
            for data_idx in epoch_iter:
                batch_g = data_queue.get()
                loss = self.get_loss(batch_g, epoch)
                if not args.deepspeed:
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 3)
                    self.optimizer.step()
                    if self.scheduler is not None:
                        self.scheduler.step()
                else:
                    self.model.backward(loss)
                    self.model.step()

                epoch_iter.set_description(f"# Epochs {epoch}: train_loss: {loss.item():.4f}")
                losses.append(loss.item())
                                 
                if args.deepspeed:
                    world_size = dist.get_world_size()
                    values = [loss.item()]
                    values = torch.tensor(values).cuda() / world_size
                    dist.all_reduce(values, op=dist.ReduceOp.SUM)
                    if dist.get_rank() == 0:
                        wandb.log({"pretrain/loss": values.item()})
                else:
                    wandb.log({"pretrain/loss": loss.item()})
                step+=1
 
            if args.deepspeed:
                client_sd["step"] = step
                client_sd["epoch"] = epoch
                self.model.save_checkpoint(ckpt_dir, client_state=client_sd)
            print_rank_0(f"# Epoch {epoch} | train_loss: {np.mean(losses):.4f}, Memory: {show_occupied_memory():.2f} MB")

    def get_loss(self, batch_g, epoch):
        args = self._args
        if args.model == "graphmae2" :
            if args.drop_edge_rate > 0:
                batch_g, targets, feats, context_nodes_dataset_id , drop_g1, drop_g2 = batch_g
                feats = feats.to(self._device)
                batch_g = batch_g.to(self._device)
                drop_g1 = drop_g1.to(self._device)
                drop_g2 = drop_g2.to(self._device)
                loss = self.model(batch_g, feats, targets, epoch, drop_g1, drop_g2)
            else:
                batch_g, targets,feats, context_nodes_dataset_id = batch_g
                feats = feats.to(self._device)
                batch_g = batch_g.to(self._device)
                loss = self.model(batch_g, feats, targets, epoch)

        elif args.model == "graphmae" :
            if args.drop_edge_rate > 0:
                batch_g, targets, feats, context_nodes_dataset_id , drop_g = batch_g
                feats = feats.to(self._device)
                batch_g = batch_g.to(self._device)
                drop_g = drop_g.to(self._device)

                if args.log_each_dataset_loss:
                    dataset_ids = np.zeros(feats.shape[0], dtype=int)
                    for idx in range(targets.shape[0]):
                        if idx+1<targets.shape[0]:
                            dataset_ids[targets[idx]:targets[idx+1]] = context_nodes_dataset_id[idx]
                        else:
                            dataset_ids[targets[idx]:] = context_nodes_dataset_id[idx]
                               
                    loss, loss_dict = self.model(batch_g, feats, drop_g, dataset_ids=dataset_ids)
                    log_dict = {}
                    for key, value in loss_dict.items():
                        log_dict["pretrain/"+args.pretrain_dataset[key]+"-loss"] = value

                    if args.deepspeed:
                        if dist.get_rank() == 0:
                            wandb.log(log_dict)
                    else:
                        wandb.log(log_dict)
                else:
                    loss = self.model(batch_g, feats, drop_g)
            else:
                batch_g, targets, feats, context_nodes_dataset_id = batch_g
                feats = feats.to(self._device)
                batch_g = batch_g.to(self._device)
                if args.log_each_dataset_loss:
                    dataset_ids = np.zeros(feats.shape[0], dtype=int)
                    for idx in range(targets.shape[0]):
                        if idx+1<targets.shape[0]:
                            dataset_ids[targets[idx]:targets[idx+1]] = context_nodes_dataset_id[idx]
                        else:
                            dataset_ids[targets[idx]:] = context_nodes_dataset_id[idx]
            
                    loss, loss_dict = self.model(batch_g, feats, dataset_ids=dataset_ids)
                    log_dict = {}
                    for key, value in loss_dict.items():
                        log_dict["pretrain/"+args.pretrain_dataset[key]+"-loss"] = value
                    
                    if args.deepspeed:
                        if dist.get_rank() == 0:
                            wandb.log(log_dict)
                    else:
                        wandb.log(log_dict)
                else:
                    loss = self.model(batch_g, feats)
        elif args.model == "grace" or args.model == "bgrl":
            if args.drop_edge_rate > 0:
                batch_g, targets, drop_feat1, drop_feat2 , context_nodes_dataset_id , drop_g1, drop_g2 = batch_g
                drop_feat1 = drop_feat1.to(self._device)
                drop_feat2 = drop_feat2.to(self._device)
                batch_g = batch_g.to(self._device)
                drop_g1 = drop_g1.to(self._device)
                drop_g2 = drop_g2.to(self._device)
            else:
                batch_g, targets,  drop_feat1, drop_feat2, context_nodes_dataset_id = batch_g
                drop_feat1 = drop_feat1.to(self._device)
                drop_feat2 = drop_feat2.to(self._device)
                batch_g = batch_g.to(self._device)
                drop_g1 = batch_g.clone()
                drop_g2 = batch_g.clone()

            if args.model == "grace":
                loss = self.model(batch_g, drop_feat1, drop_feat2, targets, drop_g1, drop_g2)
            else:
                loss = self.model(batch_g, drop_feat1, drop_feat2, drop_g1, drop_g2)
        else:
            raise ValueError    
       
        return loss

    def infer_embeddings(self):  # preparing embeddings and labels
        args = self._args
        data_queue = queue.Queue(maxsize=15)
        num_info, label_info, self._eval_dataloader = load_dataloader("eval", args.dataset, args)
        self._num_train, self._num_val, self._num_test = num_info
        self._train_label, self._val_label, self._test_label = label_info
        with torch.no_grad():
            data_thread = threading.Thread(target=data_loading_thread, args=(data_queue,self._eval_dataloader,))
            data_thread.start()
            epoch_iter = tqdm(range(len(self._eval_dataloader)))
            self.model.to(self._device)
            self.model.eval()
            embeddings = []
            #for batch in tqdm(self._eval_dataloader, desc="Infering..."):
            for idx in epoch_iter:
                batch = data_queue.get()
                batch_g, targets, _, node_idx = batch
                batch_g = batch_g.to(self._device)
                x = batch_g.ndata.pop("feat").to(self._device)
                targets = targets.to(self._device)
                batch_emb = self.model.embed(batch_g, x)[targets]
                embeddings.append(batch_emb.cpu())

        queue_size = data_queue.qsize()
        assert queue_size == 0
        self._embeddings = torch.cat(embeddings, dim=0)
        self._train_emb = self._embeddings[:self._num_train]
        self._val_emb = self._embeddings[self._num_train:self._num_train + self._num_val]
        self._test_emb = self._embeddings[self._num_train + self._num_val:]
        print(f"train embeddings:{len(self._train_emb)}")
        print(f"val embeddings  :{len(self._val_emb)}")
        print(f"test embeddings :{len(self._test_emb)}")

    def evaluate(self):
        args = self._args
        train_emb, val_emb, test_emb = self._train_emb, self._val_emb, self._test_emb
        train_label = self._train_label.to(torch.long)
        val_label = self._val_label.to(torch.long)
        test_label = self._test_label.to(torch.long)
        acc = []

        for i, seed in enumerate(args.linear_prob_seeds):
            print(f"####### Run seed {seed} for LinearProbing...")
            set_random_seed(seed)
            criterion = torch.nn.CrossEntropyLoss()
            classifier = LogisticRegression(self._train_emb.shape[1], int(train_label.max().item() + 1)).to(
                self._device)
            optimizer = create_optimizer("adam", classifier, args.lr_f, args.weight_decay_f)
            train_loader = LinearProbingDataLoader(np.arange(len(train_emb)), train_emb, train_label,
                                                   batch_size=args.batch_size_linear_prob, num_workers=args.prob_num_workers,
                                                   persistent_workers=True, shuffle=True)
            val_loader = LinearProbingDataLoader(np.arange(len(val_emb)), val_emb, val_label,
                                                 batch_size=args.batch_size_linear_prob,
                                                 num_workers=args.prob_num_workers, persistent_workers=True, shuffle=False)
            test_loader = LinearProbingDataLoader(np.arange(len(test_emb)), test_emb, test_label,
                                                  batch_size=args.batch_size_linear_prob,
                                                  num_workers=args.prob_num_workers, persistent_workers=True, shuffle=False)
            best_val_acc = 0
            best_classifier = None
            epoch_iter = tqdm(range(args.max_epoch_f)) if not args.no_verbose else range(args.max_epoch_f)
            num_no_improve = 0
            for epoch in epoch_iter:
                classifier.train()
                classifier.to(self._device)
                for batch_x, batch_label in train_loader:
                    batch_x = batch_x.to(self._device)
                    batch_label = batch_label.to(self._device)
                    pred = classifier(batch_x)
                    loss = criterion(pred, batch_label)
                    wandb.log({"LinearProbing/loss": loss.item()})
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                with torch.no_grad():
                    classifier.eval()
                    val_acc = self.eval_forward(classifier, val_loader, val_label)
                    wandb.log({"LinearProbing/valid_acc": val_acc})
                if val_acc >= best_val_acc:
                    best_val_acc = val_acc
                    best_classifier = copy.deepcopy(classifier)
                    num_no_improve = 0
                else:
                    num_no_improve = num_no_improve +1
                
                if num_no_improve>300:
                    break

                if not args.no_verbose:
                    epoch_iter.set_description(
                        f"# Epoch: {epoch}, train_loss:{loss.item(): .4f}, val_acc:{val_acc:.4f}")

            best_classifier.eval()
            with torch.no_grad():
                test_acc = self.eval_forward(best_classifier, test_loader, test_label)
                print(f"# test_acc: {test_acc:.4f}")
            acc.append(test_acc)

        print(f"# test_acc: {np.mean(acc):.4f}±{np.std(acc):.4f}")
        wandb.log({"LinearProbing/test_acc": np.mean(acc)})
        return acc

    def eval_forward(self, classifier, loader, label):
        pred_all = []
        for batch_x, _ in loader:
            batch_x = batch_x.to(self._device)
            pred = classifier(batch_x)
            pred_all.append(pred.cpu())
        pred = torch.cat(pred_all, dim=0)
        acc = accuracy(pred, label)
        return acc


    def incontext_evaluate(self):
        args = self._args
        device = args.device      
        dataset_name = args.dataset
        node_classify_task = ["ogbn-arxiv","Cora","Pubmed"]
        link_predict_task = ["FB15K237","WN18RR"]
        print(f"{args.eval_num_label}-ways, {args.eval_num_support}-shots, {args.eval_num_query}-querys, {args.khop}-khop, {args.total_steps}-total_steps")
        acc_list = []
        
        if dataset_name in link_predict_task:
            args.num_hidden = args.num_hidden*2
         
        for i, seed in enumerate(args.linear_prob_seeds):
            print(f"####### Run In-Context Evaluation #######")
            set_random_seed(seed)
            eval_dataloader = setup_incontext_dataloader(args.dataset, args)
            with torch.no_grad():
                self.model.to(device)
                self.model.eval()
                index = []
                for i in range(args.eval_num_label):
                    index.extend([i] * args.eval_num_support)
                index = torch.LongTensor(index).to(device).unsqueeze(1).expand(-1, args.num_hidden)
                accs = []
                for batch in tqdm(eval_dataloader):
                    batch = {k: v.to(device) for k, v in batch.items()}
                    data_graph = batch["data_graph"]
                    rep = self.model.embed(data_graph, data_graph.ndata["feat"].to(torch.float32))
                    batch_nodes = data_graph.batch_num_nodes()
                    batch_idx = torch.cumsum(batch_nodes, dim=0)
                    batch_idx = torch.cat([torch.LongTensor([0]).to(batch_idx.device), batch_idx[:-1]])
                    example_emb = rep[batch_idx]

                    if dataset_name in link_predict_task:
                        assert example_emb.shape[0]%2 == 0
                        even_row = example_emb[::2,:]
                        odd_row = example_emb[1::2,:]
                        example_emb = torch.cat((even_row,odd_row),dim=1)

                    label_emb = torch.zeros(args.eval_num_label, args.num_hidden).to(device)
                    label_emb = label_emb.scatter_reduce(0, index, example_emb, reduce="mean")

                    total_query = args.eval_num_support * args.eval_num_label
                    query_emb = example_emb[total_query:]

                    norm_query_emb = query_emb / torch.norm(query_emb, dim=1, keepdim=True)
                    norm_label_emb = label_emb / torch.norm(label_emb, dim=1, keepdim=True)

                    logits = torch.matmul(norm_query_emb, norm_label_emb.T)
                    acc = accuracy(logits, batch["labels"])
                    accs.append(acc)
            print(f"# acc: {np.mean(accs):.4f}±{np.std(accs):.4f}")
            acc_list = acc_list + accs
        print(f"# acc: {np.mean(acc_list):.4f}±{np.std(acc_list):.4f}")
        return acc_list