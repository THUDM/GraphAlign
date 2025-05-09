import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn
from dgl.utils import expand_as_pair
from src.gnn_modules.module_utils import create_activation,create_norm
from fmoe import SSLfmoefy
from src.utils.utils import print_rank_0

class GCN(nn.Module):
    def __init__(self,
                in_dim,
                num_hidden,
                out_dim,
                num_layers,
                dropout,
                activation,
                residual,
                norm,
                encoding=False,
                top_k=1,
                hhsize_time=1,
                num_expert=4,
                moe=False,
                moe_use_linear=False,
                moe_layer=None
                 ):
        super(GCN, self).__init__()
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.gcn_layers = nn.ModuleList()
        self.activation = activation
        self.dropout = dropout
        #lhz add
        if moe_layer==None:
            moe_layer=list(range(num_layers))

        last_activation = create_activation(activation) if encoding else None
        last_residual = encoding and residual
        last_norm = norm if encoding else None

        if num_layers == 1:
            self.gcn_layers.append(GraphConv(
                in_dim, out_dim, residual=last_residual, norm=last_norm, activation=last_activation,
                top_k=top_k, hhsize_time=hhsize_time, num_expert=num_expert, moe=moe if 0 in moe_layer else False , moe_use_linear=moe_use_linear))
        else:
            # input projection (no residual)
            self.gcn_layers.append(GraphConv(
                in_dim, num_hidden, residual=residual, norm=norm, activation=create_activation(activation),
                top_k=top_k, hhsize_time=hhsize_time, num_expert=num_expert,  moe=moe if 0 in moe_layer else False , moe_use_linear=moe_use_linear))
            # hidden layers
            for l in range(1, num_layers - 1):
                # due to multi-head, the in_dim = num_hidden * num_heads
                self.gcn_layers.append(GraphConv(
                    num_hidden, num_hidden, residual=residual, norm=norm, activation=create_activation(activation),
                    top_k=top_k, hhsize_time=hhsize_time, num_expert=num_expert,  moe=moe if l in moe_layer else False , moe_use_linear=moe_use_linear))
            # output projection
            self.gcn_layers.append(GraphConv(
                num_hidden, out_dim, residual=last_residual, activation=last_activation, norm=last_norm,
                top_k=top_k, hhsize_time=hhsize_time, num_expert=num_expert, moe=moe if num_layers-1 in moe_layer else False , moe_use_linear=moe_use_linear))

        self.norms = None
        self.head = nn.Identity()

    def forward(self, g, inputs, return_hidden=False):
        h = inputs
        hidden_list = []
        for l in range(self.num_layers):
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = self.gcn_layers[l](g, h)
            if self.norms is not None and l != self.num_layers - 1:
                h = self.norms[l](h)
            hidden_list.append(h)
        # output projection
        if self.norms is not None and len(self.norms) == self.num_layers:
            h = self.norms[-1](h)
        if return_hidden:
            return self.head(h), hidden_list
        else:
            return self.head(h)

    def reset_classifier(self, num_classes):
        self.head = nn.Linear(self.out_dim, num_classes)


class GraphConv(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 norm=None,
                 activation=None,
                 residual=True,
                 top_k=1,
                 hhsize_time=1,
                 num_expert=4,
                 moe=False,
                 moe_use_linear=False
                 ):
        super().__init__()
        self._in_feats = in_dim
        self._out_feats = out_dim

        self.fc = nn.Linear(in_dim, out_dim)

        if residual:
            if self._in_feats != self._out_feats:
                self.res_fc = nn.Linear(
                    self._in_feats, self._out_feats, bias=False)
            else:
                self.res_fc = nn.Identity()
        else:
            self.register_buffer('res_fc', None)

        if norm == "batchnorm":
            self.norm = nn.BatchNorm1d(out_dim)
        elif norm == "layernorm":
            self.norm = nn.LayerNorm(out_dim)
        else:
            self.norm = None

        self.norm = norm
        if norm is not None:
            self.norm = norm(out_dim)
        self._activation = activation

        self.reset_parameters()

        #add MOE
        if moe:
            print_rank_0("use moe bulid GCN")
            #if self.fc != None :
                #self.fc = SSLfmoefy(moe_num_experts=num_expert, hidden_size=self.fc.in_features, hidden_hidden_size=self.fc.in_features*hhsize_time ,d_outsize=self.fc.out_features, top_k=top_k, use_linear=moe_use_linear)
            #if hasattr(self, 'res_fc') and isinstance(self.res_fc, nn.Linear) :
                #self.res_fc = SSLfmoefy(moe_num_experts=num_expert, hidden_size=self.res_fc.in_features, hidden_hidden_size=self.res_fc.in_features*hhsize_time ,d_outsize=self.res_fc.out_features, top_k=top_k, use_linear=moe_use_linear)




    def reset_parameters(self):
        self.fc.reset_parameters()

    def forward_old(self, graph, feat):
        with graph.local_scope():
            aggregate_fn = fn.copy_u('h', 'm')
            # if edge_weight is not None:
            #     assert edge_weight.shape[0] == graph.number_of_edges()
            #     graph.edata['_edge_weight'] = edge_weight
            #     aggregate_fn = fn.u_mul_e('h', '_edge_weight', 'm')

            # (BarclayII) For RGCN on heterogeneous graphs we need to support GCN on bipartite.
            feat_src, feat_dst = expand_as_pair(feat, graph)
            # if self._norm in ['left', 'both']:
            degs = graph.out_degrees().float().clamp(min=1)
            norm = torch.pow(degs, -0.5)
            shp = norm.shape + (1,) * (feat_src.dim() - 1)
            norm = torch.reshape(norm, shp)
            feat_src = feat_src * norm

            # if self._in_feats > self._out_feats:
            #     # mult W first to reduce the feature size for aggregation.
            #     # if weight is not None:
            #         # feat_src = th.matmul(feat_src, weight)
            #     graph.srcdata['h'] = feat_src
            #     graph.update_all(aggregate_fn, fn.sum(msg='m', out='h'))
            #     rst = graph.dstdata['h']
            # else:
            # aggregate first then mult W
            graph.srcdata['h'] = feat_src
            graph.update_all(aggregate_fn, fn.sum(msg='m', out='h'))
            rst = graph.dstdata['h']

            rst = self.fc(rst)

            # if self._norm in ['right', 'both']:
            degs = graph.in_degrees().float().clamp(min=1)
            norm = torch.pow(degs, -0.5)
            shp = norm.shape + (1,) * (feat_dst.dim() - 1)
            norm = torch.reshape(norm, shp)
            rst = rst * norm

            if self.res_fc is not None:
                rst = rst + self.res_fc(feat_dst)

            if self.norm is not None:
                rst = self.norm(rst)

            if self._activation is not None:
                rst = self._activation(rst)

            return rst

    def forward(self, graph, feat):
        with graph.local_scope():
            aggregate_fn = fn.copy_u('h', 'm')
            feat_src, feat_dst = expand_as_pair(feat, graph)

            feat_src = self.fc(feat_src)

            degs = graph.out_degrees().float().clamp(min=1)
            norm = torch.pow(degs, -0.5)
            shp = norm.shape + (1,) * (feat_src.dim() - 1)
            norm = torch.reshape(norm, shp)
            feat_src = feat_src * norm


            graph.srcdata['h'] = feat_src
            graph.update_all(aggregate_fn, fn.sum(msg='m', out='h'))
            rst = graph.dstdata['h']

            # rst = self.fc(rst)


            degs = graph.in_degrees().float().clamp(min=1)
            norm = torch.pow(degs, -0.5)
            shp = norm.shape + (1,) * (feat_dst.dim() - 1)
            norm = torch.reshape(norm, shp)
            rst = rst * norm

            if self.res_fc is not None:
                rst = rst + self.res_fc(feat_dst)

            if self.norm is not None:
                rst = self.norm(rst)

            if self._activation is not None:
                rst = self._activation(rst)

            return rst
