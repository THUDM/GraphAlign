import torch
import torch.nn as nn
from typing import Optional
from gnn_modules import setup_module
from utils.augmentation import random_aug
import numpy as np
import torch.nn.functional as F
from dgl.nn import GraphConv


class model_cca_ssg(nn.Module):
    def __init__(
            self,
            in_dim: int,
            num_hidden: int,
            num_layers: int,
            nhead: int,
            activation: str,
            feat_drop: float,
            attn_drop: float,
            negative_slope: float,
            residual: bool,
            norm: Optional[str],
            encoder_type: str = "gat",
            der: float = 0.2,
            dfr: float = 0.2,
            lambd: float = 1e-3
    ):
        super().__init__()
        self._encoder_type = encoder_type
        self.der = der
        self.dfr = dfr
        self.lambd = lambd
        assert num_hidden % nhead == 0
        if encoder_type in ("gat", "dotgat"):
            enc_num_hidden = num_hidden // nhead
            enc_nhead = nhead
        else:
            enc_num_hidden = num_hidden
            enc_nhead = 1
        # DGL implemented encoders, encoder_type in [gcn,gat,dotgat,gin] available
        if encoder_type == 'gcn':
            self.encoder = GCN(in_dim, enc_num_hidden, enc_num_hidden, num_layers)
        else:
            self.encoder = setup_module(
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

    def embed(self, graph, feat):
        out = self.encoder(graph, feat)
        return out

    def forward(self, graph, feat):
        device = graph.device
        graph1, feat1 = random_aug(graph.remove_self_loop(), feat, self.dfr, self.der)
        graph2, feat2 = random_aug(graph.remove_self_loop(), feat, self.dfr, self.der)
        graph1 = graph1.add_self_loop()
        graph2 = graph2.add_self_loop()
        h1 = self.encoder(graph1, feat1)
        h2 = self.encoder(graph2, feat2)
        z1 = (h1 - h1.mean(0)) / (h1.std(0))
        z2 = (h2 - h2.mean(0)) / (h2.std(0))
        c = torch.mm(z1.T, z2)
        c1 = torch.mm(z1.T, z1)
        c2 = torch.mm(z2.T, z2)
        N = graph.number_of_nodes()
        c = c / N
        c1 = c1 / N
        c2 = c2 / N
        loss_inv = -torch.diagonal(c).sum()
        iden = torch.tensor(np.eye(c.shape[0])).to(device)
        loss_dec1 = (iden - c1).pow(2).sum()
        loss_dec2 = (iden - c2).pow(2).sum()
        loss = loss_inv + self.lambd * (loss_dec1 + loss_dec2)
        # print(f"loss0: {loss}")
        # raise NotImplementedError
        return loss


class LogReg(nn.Module):
    def __init__(self, hid_dim, out_dim):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        ret = self.fc(x)
        return ret

class GCN(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers):
        super().__init__()

        self.n_layers = n_layers
        self.convs = nn.ModuleList()

        self.convs.append(GraphConv(in_dim, hid_dim, norm='both'))

        if n_layers > 1:
            for i in range(n_layers - 2):
                self.convs.append(GraphConv(hid_dim, hid_dim, norm='both'))
            self.convs.append(GraphConv(hid_dim, out_dim, norm='both'))

    def forward(self, graph, x):

        for i in range(self.n_layers - 1):
            x = F.relu(self.convs[i](graph, x))
        x = self.convs[-1](graph, x)

        return x
