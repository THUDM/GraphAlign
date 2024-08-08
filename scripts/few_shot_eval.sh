Device=$1
Data_dir=$2
GNN_checkpoint_path=$3

python main_ssl_gnn_train.py \
--dataset FB15K237 \
--model graphmae \
--use_cfg \
--device $Device \
--pretrain_seed 0 \
--load_model \
--load_model_path  $GNN_checkpoint_path  \
--data_dir $Data_dir \
--moe \
--moe_use_linear \
--top_k 1  \
--num_expert 4 \
--moe_layer 0 \
--feat_type e5_float16 \
--eval_num_label 20 \
--eval_num_support  1 \
--eval_num_query 20 \
--khop  1 \
--total_steps  500 \
--sample_position total \
--fs_label ofa \
--few_shot \
--use_cfg_path ./configs/GraphMAE_configs.yml \

