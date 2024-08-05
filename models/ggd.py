import torch.nn as nn
import torch
from typing import Optional
import copy
import random
import dgl.function as fn
from sklearn import preprocessing as sk_prep
from gnn_modules import setup_module


class model_ggd(nn.Module):
    def __init__(
            self,
            in_dim: int,
            num_hidden: int,
            num_layers: int,
            nhead: int,  # 4
            activation: str,
            feat_drop: float,
            attn_drop: float,
            negative_slope: float,
            residual: bool,
            norm: Optional[str],
            encoder_type: str = "gcn",
            num_proj_layers=1,
            drop_feat=0.2
    ):
        super(model_ggd, self).__init__()
        self.drop_feat = drop_feat
        self.encoder = Encoder(
            in_dim,
            num_hidden,
            num_layers,
            nhead,  # 4
            activation,
            feat_drop,
            attn_drop,
            negative_slope,
            residual,
            norm,
            encoder_type
        )
        self.mlp = torch.nn.ModuleList()
        for i in range(num_proj_layers):
            self.mlp.append(nn.Linear(num_hidden, num_hidden))
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, g, features, corrupt_feat):
        aug_feat, aug_corrupt_feat = aug_feature_dropout(features, corrupt_feat, self.drop_feat)
        #h_1 = self.encoder(g, aug_feat, corrupt=False)
        #h_2 = self.encoder(g, aug_feat, corrupt=True)
        h_1 = self.encoder(g, aug_feat)
        h_2 = self.encoder(g, aug_corrupt_feat)
        sc_1 = h_1.squeeze(0)
        sc_2 = h_2.squeeze(0)
        for i, lin in enumerate(self.mlp):
            sc_1 = lin(sc_1)
            sc_2 = lin(sc_2)
        
        sc_1 = sc_1.sum(1).unsqueeze(0)
        sc_2 = sc_2.sum(1).unsqueeze(0)

        logits = torch.cat((sc_1, sc_2), 1)
        lbl_1 = torch.ones(1, g.num_nodes())
        lbl_2 = torch.zeros(1, g.num_nodes())
        labels = torch.cat((lbl_1, lbl_2), 1).to(logits.device)
        loss = self.loss(logits, labels)
        return loss

    # def embed(self, g, features):
    #     #h_1 = self.encoder(g, features, corrupt=False)
    #     h_1 = self.encoder(g, features)
    #     feat = h_1.clone().squeeze(0)
    #     degs = g.in_degrees().float().clamp(min=1)
    #     norm = torch.pow(degs, -0.5)
    #     norm = norm.to(h_1.device).unsqueeze(1)
    #     for _ in range(10):
    #         feat = feat * norm
    #         g.ndata['h2'] = feat
    #         g.update_all(fn.copy_u('h2', 'm'), fn.sum('m', 'h2'))
    #         feat = g.ndata.pop('h2')
    #         feat = feat * norm
    #     h_2 = feat.unsqueeze(0)
    #     embeds = (h_1 + h_2).squeeze(0).detach()
    #     embeds = sk_prep.normalize(X=embeds.cpu().numpy(), norm="l2")
    #     embeds = torch.FloatTensor(embeds).to(h_1.device)
    #     return embeds

    def embed(self, g, features):
        embeds = self.encoder(g, features)
        return embeds


class Encoder(nn.Module):
    def __init__(
            self,
            in_dim: int,
            num_hidden: int,
            num_layers: int,
            nhead: int,  # 4
            activation: str,
            feat_drop: float,
            attn_drop: float,
            negative_slope: float,
            residual: bool,
            norm: Optional[str],
            encoder_type: str = "gcn",
    ):
        super(Encoder, self).__init__()
        self._encoder_type = encoder_type
        assert num_hidden % nhead == 0
        if encoder_type in ("gat", "dotgat"):
            enc_num_hidden = num_hidden // nhead
            enc_nhead = nhead
        else:
            enc_num_hidden = num_hidden
            enc_nhead = 1
        self.conv = self.encoder = setup_module(
            m_type=encoder_type,
            enc_dec="encoding",
            in_dim=in_dim,
            num_hidden=enc_num_hidden,
            out_dim=enc_num_hidden,
            num_layers=num_layers,
            nhead=enc_nhead,
            nhead_out=enc_nhead,
            concat_out=True,
            activation=activation,
            dropout=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
        )

    # def forward(self, g, features, corrupt=False):
    #     if corrupt:
    #         perm = torch.randperm(g.number_of_nodes())
    #         features = features[perm]
    #     features = self.conv(g, features)
    #     return features

    def forward(self, g, features):
        features = self.conv(g, features)
        return features


def aug_feature_dropout(input_feat, corrupt_feat, drop_percent=0.2):
    aug_input_feat = copy.deepcopy(input_feat)
    aug_input_corrupt_feat = copy.deepcopy(corrupt_feat)
    drop_feat_num = int(aug_input_feat.shape[1] * drop_percent)
    drop_idx = random.sample([i for i in range(aug_input_feat.shape[1])], drop_feat_num)
    aug_input_feat[:, drop_idx] = 0
    aug_input_corrupt_feat[:, drop_idx] = 0
    return aug_input_feat, aug_input_corrupt_feat
