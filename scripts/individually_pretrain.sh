Device=$1
Data_dir=$2
export CUDA_VISIBLE_DEVICES=$Device

# graphmae ogbn-arxiv
python main_ssl_gnn_train.py \
--dataset ogbn-arxiv \
--model graphmae \
--use_cfg \
--device 0 \
--pretrain_seed 0  \
--data_dir $Data_dir \
--pretrain_dataset ogbn-arxiv \
--dataset_drop_edge ogbn-arxiv  \
--drop_edge_rate 0.0 \
--drop_model random \
--pretrain_num_workers 4 \
--feat_type e5_float16 \
--use_cfg_path ./configs/GraphMAE_configs.yml

#ogbn-products
# --dataset ogbn-products \
# --model graphmae \
# --use_cfg \
# --device 0 \
# --pretrain_seed 0  \
# --data_dir data_dir_path \
# --pretrain_dataset ogbn-products \
# --dataset_drop_edge ogbn-products  \
# --drop_edge_rate 0.0 \
# --drop_model random \
# --pretrain_num_workers 4 \
# --feat_type e5_float16 \
# --use_cfg_path ./configs/GraphMAE_configs.yml
    
#ogbn-papers100M
# --dataset ogbn-papers100M \
# --model graphmae \
# --use_cfg \
# --device 0 \
# --pretrain_seed 0  \
# --data_dir data_dir_path \
# --pretrain_dataset ogbn-papers100M \
# --dataset_drop_edge ogbn-papers100M  \
# --drop_edge_rate 0.0 \
# --drop_model random \
# --pretrain_num_workers 4 \
# --feat_type e5_float16 \
# --use_cfg_path ./configs/GraphMAE_configs.yml


















