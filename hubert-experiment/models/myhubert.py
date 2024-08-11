import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

from fairseq.data.dictionary import Dictionary
from fairseq.tasks.hubert_pretraining import HubertPretrainingConfig
from fairseq.models import register_model
from fairseq.models.hubert import HubertConfig, HubertModel
from fairseq.tasks.hubert_pretraining import HubertPretrainingTask
from fairseq.utils import buffered_arange


logger = logging.getLogger(__name__)


@dataclass
class MyHubertConfig(HubertConfig):
    # negative selection
    num_negatives: int = field(
        default=100,
        metadata={"help": "number of negative examples from the same sample"},
    )
    # negatives_from_everywhere: bool = field(
    #     default=False,
    #     metadata={"help": "sample negatives from everywhere, not just masked states"},
    # )
    cross_sample_negatives: int = field(
        default=0, metadata={"help": "number of negative examples from the any sample"}
    )
    # codebook_negatives: int = field(
    #     default=0, metadata={"help": "number of negative examples codebook"}
    # )


@register_model("myhubert", dataclass=MyHubertConfig)
class MyHubertModel(HubertModel):
    def __init__(
        self,
        cfg: HubertConfig,
        task_cfg: HubertPretrainingConfig,
        dictionaries: List[Dictionary],
    ) -> None:
        super().__init__(cfg, task_cfg, dictionaries)

        self.n_negatives = cfg.num_negatives
        self.cross_sample_negatives = cfg.cross_sample_negatives
        # self.codebook_negatives = cfg.codebook_negatives
        # self.negatives_from_everywhere = cfg.negatives_from_everywhere

    @classmethod
    def build_model(cls, cfg: HubertConfig, task: HubertPretrainingTask):
        model = MyHubertModel(cfg, task.cfg, task.dictionaries)
        return model

    def sample_negatives(
        self, y: torch.Tensor, num_frames: int, padding_count: Optional[int] = None
    ):
        if self.n_negatives == 0 and self.cross_sample_negatives == 0:
            return y.new(0)

        bsz, tsz, fsz = y.shape
        y = y.view(-1, fsz)  # BTC => (BxT)C

        # FIXME: what happens if padding_count is specified?
        cross_high = tsz * bsz
        high = tsz - (padding_count or 0)
        with torch.no_grad():
            assert high > 1, f"{bsz,tsz,fsz}"

            if self.n_negatives > 0:
                tszs = (
                    buffered_arange(num_frames)
                    .unsqueeze(-1)
                    .expand(-1, self.n_negatives)
                    .flatten()
                )

                neg_idxs = torch.randint(
                    low=0, high=high - 1, size=(bsz, self.n_negatives * num_frames)
                )
                neg_idxs[neg_idxs >= tszs] += 1

            if self.cross_sample_negatives > 0:
                tszs = (
                    buffered_arange(num_frames)
                    .unsqueeze(-1)
                    .expand(-1, self.cross_sample_negatives)
                    .flatten()
                )

                cross_neg_idxs = torch.randint(
                    low=0,
                    high=cross_high - 1,
                    size=(bsz, self.cross_sample_negatives * num_frames),
                )
                cross_neg_idxs[cross_neg_idxs >= tszs] += 1

        if self.n_negatives > 0:
            neg_idxs = neg_idxs + (torch.arange(bsz).unsqueeze(1) * high)
        else:
            neg_idxs = cross_neg_idxs

        if self.cross_sample_negatives > 0 and self.n_negatives > 0:
            neg_idxs = torch.cat([neg_idxs, cross_neg_idxs], dim=1)

        negs = y[neg_idxs.view(-1)]
        negs = negs.view(
            bsz, num_frames, self.n_negatives + self.cross_sample_negatives, fsz
        ).permute(
            2, 0, 1, 3
        )  # to NxBxTxC
        return negs, neg_idxs

    def forward(
        self,
        source: torch.Tensor,
        target_list: Optional[List[torch.Tensor]] = None,
        padding_mask: Optional[torch.Tensor] = None,
        mask: bool = True,
        features_only: bool = False,
        output_layer: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """output layer is 1-based"""
        features = self.forward_features(source)
        if target_list is not None:
            features, target_list = self.forward_targets(features, target_list)

        features_pen = features.float().pow(2).mean()

        features = features.transpose(1, 2)
        features = self.layer_norm(features)
        unmasked_features = features.clone()

        if padding_mask is not None:
            padding_mask = self.forward_padding_mask(features, padding_mask)

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        features = self.dropout_input(features)
        unmasked_features = self.dropout_features(unmasked_features)

        if mask:
            x, mask_indices = self.apply_mask(features, padding_mask, target_list)
        else:
            x = features
            mask_indices = None

        # feature: (B, T, D), float
        # target: (B, T), long
        # x: (B, T, D), float
        # padding_mask: (B, T), bool
        # mask_indices: (B, T), bool
        x, layer_results = self.encoder(
            x,
            padding_mask=padding_mask,
            layer=None if output_layer is None else output_layer - 1,
        )

        if features_only:
            return {
                "x": x,
                "padding_mask": padding_mask,
                "features": features,
                "layer_results": layer_results,
            }

        def compute_pred(proj_x, target, label_embs):
            # compute logits for the i-th label set
            y = torch.index_select(label_embs, 0, target.long())
            negs = label_embs.unsqueeze(1).expand(-1, proj_x.size(0), -1)
            if self.target_glu:
                y = self.target_glu(y)
                negs = self.target_glu(negs)
            # proj_x: (S, D)
            # y: (S, D)
            # negs: (Neg, S, D)
            return self.compute_nce(proj_x, y, negs)

        label_embs_list = self.label_embs_concat.split(self.num_classes, 0)

        if not self.skip_masked:
            masked_indices = torch.logical_and(~padding_mask, mask_indices)
            proj_x_m = self.final_proj(x[masked_indices])
            if self.untie_final_proj:
                proj_x_m_list = proj_x_m.chunk(len(target_list), dim=-1)
            else:
                proj_x_m_list = [proj_x_m for _ in range(len(target_list))]

            # logit_m_list = [
            #     compute_pred(proj_x_m, t[masked_indices], label_embs_list[i])
            #     for i, (proj_x_m, t) in enumerate(zip(proj_x_m_list, target_list))
            # ]

            # label embeddings of masked frames
            y_list = [
                torch.index_select(embs, 0, t[masked_indices]).view(
                    x.size(0), -1, embs.size(-1)
                )
                for t, embs in zip(target_list, label_embs_list)
            ]
            negs_list = [self.sample_negatives(y, y.size(1))[0] for y in y_list]
            transf = self.target_glu or (lambda _: _)
            logit_m_list = [
                self.compute_nce(
                    proj_x_m,
                    transf(y.view(-1, y.size(-1))),
                    transf(negs.view(negs.size(0), -1, negs.size(-1))),
                )
                for proj_x_m, y, negs in zip(proj_x_m_list, y_list, negs_list)
            ]

        else:
            logit_m_list = [None for _ in target_list]

        if not self.skip_nomask:
            nomask_indices = torch.logical_and(~padding_mask, ~mask_indices)
            proj_x_u = self.final_proj(x[nomask_indices])
            if self.untie_final_proj:
                proj_x_u_list = proj_x_u.chunk(len(target_list), dim=-1)
            else:
                proj_x_u_list = [proj_x_u for _ in range(len(target_list))]

            logit_u_list = [
                compute_pred(proj_x_u, t[nomask_indices], label_embs_list[i])
                for i, (proj_x_u, t) in enumerate(zip(proj_x_u_list, target_list))
            ]
        else:
            logit_u_list = [None for _ in target_list]

        result = {
            "logit_m_list": logit_m_list,
            "logit_u_list": logit_u_list,
            "padding_mask": padding_mask,
            "features_pen": features_pen,
        }
        return result
