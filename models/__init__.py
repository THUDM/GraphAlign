from models.graphmae import model_graphmae
from models.graphmae2 import model_graphmae2
from models.grace import model_grace
from models.cca_ssg import model_cca_ssg
from models.bgrl import model_bgrl
from models.ggd import model_ggd
from gnn_modules import Supervised_gnn_classification #,LightGCN
import torch


def build_model(args):
    if args.model == 'graphmae':
        return model_graphmae(
            in_dim=args.num_features,
            num_hidden=args.num_hidden,
            num_layers=args.num_layers,
            nhead=args.num_heads,
            nhead_out=args.num_out_heads,
            activation=args.activation,
            feat_drop=args.in_drop,
            attn_drop=args.attn_drop,
            negative_slope=args.negative_slope,
            residual=args.residual,
            encoder_type=args.encoder,
            decoder_type=args.decoder,
            mask_rate=args.mask_rate,
            norm=args.norm,
            loss_fn=args.loss_fn,
            drop_edge_rate=args.drop_edge_rate,
            replace_rate=args.replace_rate,
            alpha_l=args.alpha_l,
            concat_hidden=args.concat_hidden,
            top_k=args.top_k,
            hhsize_time=args.hiddenhidden_size_times,
            num_expert=args.num_expert,
            moe=args.moe,
            moe_use_linear=args.moe_use_linear,
            decoder_no_moe=args.decoder_no_moe, 
            moe_layer=args.moe_layer
        )
    elif args.model == 'graphmae2':
        return model_graphmae2(
            in_dim=args.num_features,
            num_hidden=args.num_hidden,
            num_layers=args.num_layers,
            num_dec_layers=args.num_dec_layers,
            num_remasking=args.num_remasking,
            nhead=args.num_heads,
            nhead_out=args.num_out_heads,
            activation=args.activation,
            feat_drop=args.in_drop,
            attn_drop=args.attn_drop,
            negative_slope=args.negative_slope,
            residual=args.residual,
            encoder_type=args.encoder,
            decoder_type=args.decoder,
            mask_rate=args.mask_rate,
            remask_rate=args.remask_rate,
            mask_method=args.mask_method,
            norm=args.norm,
            loss_fn=args.loss_fn,
            drop_edge_rate=args.drop_edge_rate,
            alpha_l=args.alpha_l,
            lam=args.lam,
            delayed_ema_epoch=args.delayed_ema_epoch,
            replace_rate=args.replace_rate,
            remask_method=args.remask_method,
            momentum=args.momentum,
            zero_init=args.dataset in ("cora", "pubmed", "citeseer"),
            top_k=args.top_k,
            hhsize_time=args.hiddenhidden_size_times,
            num_expert=args.num_expert,
            moe=args.moe,
            moe_use_linear=args.moe_use_linear,
            decoder_no_moe=args.decoder_no_moe, 
            moe_layer=args.moe_layer,
            deepspeed=args.deepspeed,
            graphmae2_ema_graph_nodrop=args.graphmae2_ema_graph_nodrop
        )
    elif args.model == 'cca_ssg':
        return model_cca_ssg(
            in_dim=args.num_features,
            num_hidden=args.num_hidden,
            num_layers=args.num_layers,
            nhead=args.num_heads,
            activation=args.activation,
            feat_drop=args.in_drop,
            attn_drop=args.attn_drop,
            negative_slope=args.negative_slope,
            residual=args.residual,
            norm=args.norm,
            encoder_type=args.encoder,
            der=args.der,
            dfr=args.dfr,
            lambd=args.lambd
        )
    elif args.model == 'grace':
        return model_grace(
            in_dim=args.num_features,
            num_hidden=args.num_hidden,
            num_proj_hidden=args.num_proj_hidden,
            num_layers=args.num_layers,
            nhead=args.num_heads,
            activation=args.activation,
            feat_drop=args.in_drop,
            attn_drop=args.attn_drop,
            negative_slope=args.negative_slope,
            residual=args.residual,
            norm=args.norm,
            encoder_type=args.encoder,
            drop_edge_rate_1=args.drop_edge_rate_1,
            drop_edge_rate_2=args.drop_edge_rate_2,
            drop_feature_rate_1=args.drop_feature_rate_1,
            drop_feature_rate_2=args.drop_feature_rate_2,
            tau=args.tau,
            top_k=args.top_k,
            hhsize_time=args.hiddenhidden_size_times,
            num_expert=args.num_expert,
            moe=args.moe,
            moe_use_linear=args.moe_use_linear,
            decoder_no_moe=args.decoder_no_moe, 
            moe_layer=args.moe_layer,
            deepspeed=args.deepspeed,
            graphmae2_ema_graph_nodrop=args.graphmae2_ema_graph_nodrop
        )
    elif args.model == 'bgrl':
        return model_bgrl(
            in_dim=args.num_features,
            num_hidden=args.num_hidden,
            num_layers=args.num_layers,
            nhead=args.num_heads,
            activation=args.activation,
            feat_drop=args.in_drop,
            attn_drop=args.attn_drop,
            negative_slope=args.negative_slope,
            residual=args.residual,
            norm=args.norm,
            encoder_type=args.encoder,
            pred_hid=args.pred_hid,
            moving_average_decay=args.moving_average_decay,
            ema_total_steps=args.ema_total_steps,
            drop_edge_rate_1=args.drop_edge_rate_1,
            drop_edge_rate_2=args.drop_edge_rate_2,
            drop_feature_rate_1=args.drop_feature_rate_1,
            drop_feature_rate_2=args.drop_feature_rate_2,
            graphmae2_ema_graph_nodrop=args.graphmae2_ema_graph_nodrop
        )
    elif args.model == 'ggd':
        return model_ggd(
            in_dim=args.num_features,
            num_hidden=args.num_hidden,
            num_layers=args.num_layers,
            nhead=args.num_heads,
            activation=args.activation,
            feat_drop=args.in_drop,
            attn_drop=args.attn_drop,
            negative_slope=args.negative_slope,
            residual=args.residual,
            norm=args.norm,
            encoder_type=args.encoder,
            num_proj_layers=args.num_proj_layers,
            drop_feat=args.drop_feat
        )
    else:
        assert False and "Invalid model"


