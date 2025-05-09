from functools import partial
import torch
import torch.nn as nn
from src.gnn_modules.gin import GIN
from src.gnn_modules.gat import GAT
from src.gnn_modules.gcn import GCN
from src.gnn_modules.dot_gat import DotGAT
#from src.gnn_modules.LightGCN import LightGCN
from src.gnn_modules.module_utils import create_activation, create_norm
#from transformers import RobertaTokenizer, RobertaConfig, RobertaModel

debug = True

def setup_module(m_type, enc_dec, in_dim, num_hidden, out_dim, num_layers, dropout=0.2, activation='relu',
                 residual=False, norm=None, nhead=4,
                 nhead_out=1, attn_drop=0.1, negative_slope=0.2, concat_out=True, allow_zero_degree=True, top_k=1, hhsize_time=1, num_expert=4, moe=False, moe_use_linear=False, moe_layer=None) -> nn.Module:

    if m_type == "gat":
        mod = GAT(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            nhead=nhead,
            nhead_out=nhead_out,
            concat_out=concat_out,
            activation=activation,
            feat_drop=dropout,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=create_norm(norm),
            encoding=(enc_dec == "encoding"),
            allow_zero_degree=allow_zero_degree,
            top_k=top_k,
            hhsize_time=hhsize_time,
            num_expert=num_expert,
            moe=moe,
            moe_use_linear=moe_use_linear,
            moe_layer=moe_layer

        )
    elif m_type == "dotgat":
        mod = DotGAT(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            nhead=nhead,
            nhead_out=nhead_out,
            concat_out=concat_out,
            activation=activation,
            feat_drop=dropout,
            attn_drop=attn_drop,
            residual=residual,
            norm=create_norm(norm),
            encoding=(enc_dec == "encoding"),
            allow_zero_degree=allow_zero_degree
        )
    elif m_type == "gin":
        mod = GIN(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation,
            residual=residual,
            norm=norm,
            encoding=(enc_dec == "encoding"),
        )
    elif m_type == "gcn":
        mod = GCN(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation,
            residual=residual,
            norm=create_norm(norm),
            encoding=(enc_dec == "encoding"),
            top_k=top_k,
            hhsize_time=hhsize_time,
            num_expert=num_expert,
            moe=moe,
            moe_use_linear=moe_use_linear,
            moe_layer=moe_layer
        )
    elif m_type == "mlp":
        # * just for decoder
        mod = nn.Sequential(
            nn.Linear(in_dim, num_hidden),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Linear(num_hidden, out_dim)
        )
    elif m_type == "linear":
        mod = nn.Linear(in_dim, out_dim)
    else:
        raise NotImplementedError

    return mod

'''
class Supervised_gnn_classification(nn.Module):
    def __init__(self, m_type, in_dim, num_hidden, out_dim, num_layers, dropout=0.2, activation='relu', residual=False,
                 norm=None, nhead=4, attn_drop=0.1, negative_slope=0.2, use_roberta=False):
        super().__init__()
        self.use_roberta = use_roberta
        if self.use_roberta:
            self.ro_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
            self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.m_type = m_type
        if m_type in ("gat", "dotgat", "gin", "gcn"):
            if m_type in ("gat", "dotgat"):
                enc_num_hidden = num_hidden // nhead
                enc_nhead = nhead
            else:
                enc_num_hidden = num_hidden
                enc_nhead = 1
            self.encoder = setup_module(
                m_type=m_type,
                enc_dec="encoding",
                in_dim=in_dim,
                num_hidden=enc_num_hidden,
                out_dim=enc_num_hidden,
                num_layers=num_layers,
                nhead=enc_nhead,
                nhead_out=enc_nhead,
                concat_out=True,
                activation=activation,
                dropout=dropout,
                attn_drop=attn_drop,
                negative_slope=negative_slope,
                residual=residual,
                norm=norm
            )
            self.classifier = torch.nn.Linear(num_hidden, out_dim)
        elif m_type == "mlp":
            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(in_dim, num_hidden),
                torch.nn.PReLU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(num_hidden, out_dim)
            )
        else:
            raise NotImplementedError

    def forward(self, graph, x):
        if hasattr(self, 'roberta'):
            batch_size = 64
            encoded = []
            for i in range(len(x) // batch_size):
                batch = x[i * batch_size:(i + 1) * batch_size if (i + 1) * batch_size > len(x) - 1 else len(x)]
                encoded_input = self.ro_tokenizer(batch, padding='longest', return_tensors='pt')
                for key in encoded_input.keys():
                    encoded_input[key] = encoded_input[key].cuda()
                out = torch.mean(self.roberta(**encoded_input)[0], dim=1).squeeze()
                encoded.append(out)
            x = torch.tensor(encoded)
            print(x.shape)
        if self.m_type in ("gat", "dotgat", "gin", "gcn"):
            out = self.encoder(graph, x)
            out = self.classifier(out)
        elif self.m_type == "mlp":
            out = self.classifier(x)
        return out


class Supervised_gnn_link_prediction(nn.Module):
    def __init__(self, m_type, in_dim, num_hidden, out_dim, num_layers, dropout=0.2, activation='relu', residual=False,
                 norm=None, nhead=4, attn_drop=0.1, negative_slope=0.2, use_roberta=False):
        super().__init__()
        self.use_roberta = use_roberta
        if self.use_roberta:
            self.ro_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
            self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.m_type = m_type
        if m_type in ("gat", "dotgat", "gin", "gcn"):
            if m_type in ("gat", "dotgat"):
                enc_num_hidden = num_hidden // nhead
                enc_nhead = nhead
            else:
                enc_num_hidden = num_hidden
                enc_nhead = 1
            self.encoder = setup_module(
                m_type=m_type,
                enc_dec="encoding",
                in_dim=in_dim,
                num_hidden=enc_num_hidden,
                out_dim=enc_num_hidden,
                num_layers=num_layers,
                nhead=enc_nhead,
                nhead_out=enc_nhead,
                concat_out=True,
                activation=activation,
                dropout=dropout,
                attn_drop=attn_drop,
                negative_slope=negative_slope,
                residual=residual,
                norm=norm
            )
            self.classifier = torch.nn.Linear(num_hidden, out_dim)
        elif m_type == "mlp":
            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(in_dim, num_hidden),
                torch.nn.PReLU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(num_hidden, out_dim)
            )
        else:
            raise NotImplementedError

    def forward(self, graph, x):
        if hasattr(self, 'roberta'):
            batch_size = 64
            encoded = []
            for i in range(len(x) // batch_size):
                batch = x[i * batch_size:(i + 1) * batch_size if (i + 1) * batch_size > len(x) - 1 else len(x)]
                encoded_input = self.ro_tokenizer(batch, padding='longest', return_tensors='pt')
                for key in encoded_input.keys():
                    encoded_input[key] = encoded_input[key].cuda()
                out = torch.mean(self.roberta(**encoded_input)[0], dim=1).squeeze()
                encoded.append(out)
            x = torch.tensor(encoded)
            print(x.shape)
        if self.m_type in ("gat", "dotgat", "gin", "gcn"):
            out = self.encoder(graph, x)
            out = self.classifier(out)
        elif self.m_type == "mlp":
            out = self.classifier(x)
        return out
'''
