# eval graphmae(graphalign) checkpoint on different dataset. Change --dataset to eval different datasets ogbn-arxiv, ogbn-products, ogbn-papers100M
python main_ssl_gnn_train.py \
--dataset ogbn-arxiv \
--model graphmae \
--use_cfg \
--device 0 \
--pretrain_seed 0 \
--load_model \
--load_model_path  checkpoint_path \
--data_dir data_dir_path \
--feat_type e5_float16 \
--use_cfg_path ./configs/GraphMAE_configs.yml \
--prob_num_workers 4 \
--moe \
--moe_use_linear \
--top_k 1  \
--num_expert 4 \
--hiddenhidden_size_times 1 \
--moe_layer 0  \
--decoder_no_moe \

