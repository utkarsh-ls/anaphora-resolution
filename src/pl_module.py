from itertools import combinations
import pytorch_lightning as pl
from model import MentionModel, PairScoreModel
import torch


class PLModuleMention(pl.LightningModule):
    def __init__(self, max_seq_len, lr=1e-3, pos_wt=1e2):
        super(PLModuleMention, self).__init__()
        self.max_seq_len = max_seq_len
        self.pos_wt = pos_wt
        self.lr = lr
        self.model = MentionModel()
        self.l1_loss = torch.nn.L1Loss()
        self.loss_fn = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(self.pos_wt).to(self.device)
        )
        self.save_hyperparameters()

    def forward(self, ids, mask):
        return self.model.forward(ids, mask)

    def _common_step(self, batch, mode: str):
        tok_id_list, mask, is_mention_list, first_token_idx_list = batch
        mention_logits, word_embeds = self.forward(tok_id_list, mask)

        loss = self.loss_fn(mention_logits, is_mention_list)

        self.log_dict(
            {
                f"{mode}_loss": loss.detach(),
                f"{mode}_mt_acc": self._mention_acc(
                    mention_logits.detach(), is_mention_list, 1
                ),
                f"{mode}_no_mt_acc": self._mention_acc(
                    mention_logits.detach(), is_mention_list, 0
                ),
                f"{mode}_all_acc": self._mention_acc(
                    mention_logits.detach(), is_mention_list, -1
                ),
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return {"loss": loss, "embed": word_embeds}

    def _mention_acc(self, pred, gt, for_ment=0):
        # gt: B, L
        pred_cls = pred >= 0.5
        true_cls = gt != 0
        mask = gt == for_ment
        if for_ment == -1:
            mask = mask | True
        correct = (pred_cls == true_cls) * mask
        return correct.sum() / mask.sum()

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._common_step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer, **self.config["lr_sched"]
        # )
        return {
            "optimizer": optimizer,
            # "lr_scheduler": {
            #     "scheduler": lr_scheduler,
            #     "monitor": "val_loss",
            # },
        }


class PLModulePairScore(pl.LightningModule):
    def __init__(self, mention_wt_path: str, max_seq_len, lr=1e-3, pos_wt=1):
        super(PLModulePairScore, self).__init__()
        self.max_seq_len = max_seq_len
        self.pos_wt = pos_wt
        self.lr = lr
        self.model = PairScoreModel()
        self.bce = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(self.pos_wt).to(self.device)
        )
        self.sigmoid = torch.nn.Sigmoid()
        self.save_hyperparameters()

    def forward(self, word1_embeds, word2_embeds):
        return self.model(word1_embeds, word2_embeds)

    def loss_fn(self, mentn_cluster_lists, all_scores, selected_words):
        total_loss = 0.0
        total_samples = 0
        flattened_gt = []
        flattened_pred = []
        for sel_words, scores, mcluster_list in zip(
            selected_words, all_scores, mentn_cluster_lists
        ):
            indices = torch.arange(len(sel_words))[sel_words]
            for (i, j), score in zip(combinations(indices, 2), scores):
                is_same_cluster = torch.tensor(0, dtype=score.dtype).to(score.device)
                for c in mcluster_list:
                    if i in c and j in c:
                        is_same_cluster += 1
                        break
                total_samples += 1
                total_loss += self.BCELogitsLoss(score, is_same_cluster)
                flattened_gt.append(is_same_cluster)
                flattened_pred.append(score)
        return (
            total_loss / total_samples,
            torch.as_tensor(flattened_pred),
            torch.as_tensor(flattened_gt),
        )

    def _common_step(self, batch, mode: str):
        word1_embeds, word2_embeds, is_same_cluster_list = batch
        scores = self.forward(word1_embeds, word2_embeds)

        loss = self.bce(scores, is_same_cluster_list)

        self.log_dict(
            {
                f"{mode}_loss": loss.detach(),
                f"{mode}_pos_acc": self._pair_score_acc(
                    scores, is_same_cluster_list, 1, mode
                ),
                f"{mode}_neg_acc": self._pair_score_acc(
                    scores, is_same_cluster_list, 0, mode
                ),
                f"{mode}_all_acc": self._pair_score_acc(
                    scores, is_same_cluster_list, -1, mode
                ),
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return {"loss": loss}

    def _pair_score_acc(self, pred, gt, _class=0, mode="Train"):
        # gt: B, L
        pred_cls = self.sigmoid(pred) >= 0.5
        true_cls = gt != 0
        mask = gt == _class
        if _class == -1:
            mask = mask | True
        correct = (pred_cls == true_cls) * mask
        self.print(f"{mode}: {correct.sum()}, {mask.sum()}, {_class}")
        self.print(pred.mean().item(), pred.std().item())
        return correct.sum() / (mask.sum() + 1e-6)

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._common_step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer, **self.config["lr_sched"]
        # )
        return {
            "optimizer": optimizer,
            # "lr_scheduler": {
            #     "scheduler": lr_scheduler,
            #     "monitor": "val_loss",
            # },
        }
