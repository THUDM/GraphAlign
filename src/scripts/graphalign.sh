Device=$1
Data_dir=$2
MASTER_PORT=$(shuf -n 1 -i 10000-65535)
echo "Using deepspeed"
#localhost: gpu1,gpu2,gpu3,..

# graphalign GraphMAE on three dataset
deepspeed --include=localhost:$Device --master_port ${MASTER_PORT} main_ssl_gnn_train.py --deepspeed --deepspeed_config ./configs/ds_config.json \
--dataset graphalign \
--model graphmae \
--use_cfg \
--device 0 \
--pretrain_seed 0 \
--data_dir $Data_dir \
--pretrain_dataset ogbn-arxiv  ogbn-products ogbn-papers100M   \
--lr  0.0002  \
--weight_decay 0.04 \
--batch_size 512  \
--max_epoch 20 \
--pretrain_num_workers 4 \
--dataset_drop_edge ogbn-arxiv   ogbn-papers100M   \
--drop_edge_rate 0.5 \
--drop_model directed_to_undirected \
--moe \
--moe_use_linear \
--top_k 1  \
--num_expert 4 \
--moe_layer 0  \
--decoder_no_moe \
--default_dataset products \
--weight 8 30 15  \
--use_cfg_path ./configs/GraphMAE_configs.yml



#graphalign GraphMAE2 on three dataset
# --dataset graphalign \
# --model graphmae2 \
# --use_cfg \
# --device 0 \
# --pretrain_seed 0 \
# --data_dir data_dir_path \
# --pretrain_dataset ogbn-arxiv  ogbn-products ogbn-papers100M   \
# --lr  0.0002  \
# --weight_decay 0.04 \
# --batch_size 512  \
# --max_epoch 20 \
# --pretrain_num_workers 4 \
# --dataset_drop_edge ogbn-arxiv   ogbn-papers100M   \
# --drop_edge_rate 0.5 \
# --drop_model directed_to_undirected \
# --moe \
# --moe_use_linear \
# --top_k 1  \
# --num_expert 4 \
# --moe_layer 0  \
# --decoder_no_moe \
# --default_dataset products \
# --weight 8 30 15  \
# --use_cfg_path ./configs/GraphMAE2_configs.yml
# --graphmae2_ema_graph_nodrop


#grace on three dataset
# --dataset graphalign \
# --model grace \
# --use_cfg \
# --device 0 \
# --pretrain_seed 0 \
# --data_dir data_dir_path \
# --pretrain_dataset ogbn-arxiv  ogbn-products ogbn-papers100M   \
# --lr  0.0002  \
# --weight_decay 0.04 \
# --batch_size 512  \
# --max_epoch 20 \
# --pretrain_num_workers 4 \
# --dataset_drop_edge ogbn-arxiv   ogbn-papers100M   \
# --drop_edge_rate 0.5 \
# --drop_model directed_to_undirected \
# --dataset_drop_feat  ogbn-products \
# --drop_feature_rate_1  0.2 \
# --drop_feature_rate_2  0.4 \
# --moe \
# --moe_use_linear \
# --top_k 1  \
# --num_expert 4 \
# --moe_layer 0  \
# --default_dataset products \
# --weight 8 30 15  \
# --use_cfg_path ./configs/Grace_configs.yml


# bgrl on three dataset
# --dataset graphalign \
# --model bgrl \
# --use_cfg \
# --device 0 \
# --pretrain_seed 0 \
# --data_dir data_dir_path \
# --pretrain_dataset ogbn-arxiv  ogbn-products ogbn-papers100M   \
# --lr  0.0002  \
# --weight_decay 0.04 \
# --batch_size 512  \
# --max_epoch 10 \
# --pretrain_num_workers 4 \
# --dataset_drop_edge ogbn-arxiv   ogbn-papers100M   \
# --drop_edge_rate 0.5 \
# --drop_model directed_to_undirected \
# --dataset_drop_feat  ogbn-products \
# --drop_feature_rate_1  0.2 \
# --drop_feature_rate_2  0.2 \
# --moe \
# --moe_use_linear \
# --top_k 1  \
# --num_expert 4 \
# --moe_layer 0  \
# --default_dataset products \
# --weight 8 30 15  \
# --use_cfg_path ./configs/BGRL_configs.yml