import torch
import torch.nn.functional as F
import copy
from typing import Optional
from src.gnn_modules import setup_module
from src.utils.augmentation import random_aug, drop_feature
import torch.distributed as dist
import diffdist

class model_grace(torch.nn.Module):
    def __init__(
            self,
            in_dim: int,
            num_hidden: int,
            num_proj_hidden: int,
            num_layers: int,
            nhead: int,  # 4
            activation: str,
            feat_drop: float,
            attn_drop: float,
            negative_slope: float,
            residual: bool,
            norm: Optional[str],
            encoder_type: str = "gat",
            drop_edge_rate_1: float = 0.5,
            drop_edge_rate_2: float = 0.5,
            drop_feature_rate_1: float = 0.5,
            drop_feature_rate_2: float = 0.5,
            tau: float = 1e-3,
            top_k=1,
            hhsize_time=1,
            num_expert=4,
            moe=False,
            moe_use_linear=False,
            decoder_no_moe=False,
            moe_layer=None,
            deepspeed=False,
            graphmae2_ema_graph_nodrop=False
    ):
        super(model_grace, self).__init__()

        self.distributed = deepspeed
        self.graphmae2_ema_graph_nodrop = graphmae2_ema_graph_nodrop

        self._encoder_type = encoder_type
        assert num_hidden % nhead == 0
        if encoder_type in ("gat", "dotgat"):
            enc_num_hidden = num_hidden // nhead
            enc_nhead = nhead
        else:
            enc_num_hidden = num_hidden
            enc_nhead = 1
        # DGL implemented encoders, encoder_type in [gcn,gat,dotgat,gin] available
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
            top_k=top_k,
            hhsize_time=hhsize_time,
            num_expert=num_expert,
            moe=moe,
            moe_use_linear=moe_use_linear,
            moe_layer=moe_layer
        )
        self.drop_edge_rate_1 = drop_edge_rate_1
        self.drop_edge_rate_2 = drop_edge_rate_2
        self.drop_feature_rate_1 = drop_feature_rate_1
        self.drop_feature_rate_2 = drop_feature_rate_2
        self.tau = tau
        self.fc1 = torch.nn.Linear(num_hidden, num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, num_hidden)

    def forward(self, g, drop_feat1, drop_feat2, target: torch.Tensor, drop_g1, drop_g2):
        #graph1, feat1 = random_aug(g.remove_self_loop(), x, self.drop_feature_rate_1, self.drop_edge_rate_1)
        #graph2, feat2 = random_aug(g.remove_self_loop(), x, self.drop_feature_rate_2, self.drop_edge_rate_2)
        #graph_1 = graph1.add_self_loop()
        #graph_2 = graph2.add_self_loop()


        #feat1 = drop_feature(x, self.drop_feature_rate_1)
        #feat2 = drop_feature(x, self.drop_feature_rate_2)
        feat1 = drop_feat1
        feat2 = drop_feat2
        graph_1 = drop_g1
        if self.graphmae2_ema_graph_nodrop:
            graph_2 = g
        else:
            graph_2 = drop_g2
        #graph_2 = drop_g2

        z1 = self.embed(graph_1, feat1)
        z2 = self.embed(graph_2, feat2)
        #return self.loss(z1, z2)
        return self.loss(z1[target], z2[target])

    def embed(self, g, x: torch.Tensor):
        return self.encoder(g, x)

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))
        return -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor,
                          batch_size: int):
        # Space complexity: O(BN) (semi_loss: O(N^2))
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
            between_sim = f(self.sim(z1[mask], z2))  # [B, N]

            losses.append(-torch.log(
                between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                / (refl_sim.sum(1) + between_sim.sum(1)
                   - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))
        return torch.cat(losses)

    def loss(self, z1: torch.Tensor, z2: torch.Tensor,
             mean: bool = True, batch_size: int = 0):
        h1 = self.projection(z1)
        h2 = self.projection(z2)

        #test
        if self.distributed:
            h1_list = [torch.zeros_like(h1) for _ in range(dist.get_world_size())]
            h2_list = [torch.zeros_like(h2) for _ in range(dist.get_world_size())]
            h1_list = diffdist.functional.all_gather(h1_list, h1)
            h2_list = diffdist.functional.all_gather(h2_list, h2)
            h1 = torch.cat(h1_list, dim=0)
            h2 = torch.cat(h2_list, dim=0)
        #test


        if batch_size == 0:
            l1 = self.semi_loss(h1, h2)
            l2 = self.semi_loss(h2, h1)
        else:
            l1 = self.batched_semi_loss(h1, h2, batch_size)
            l2 = self.batched_semi_loss(h2, h1, batch_size)

        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()
        return ret


