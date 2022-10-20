import pytorch_lightning as pl
from model import MentionModel
import torch


class PLModule(pl.LightningModule):
    def __init__(self, max_seq_len, lr=1e-3, pos_wt=1e2):
        super(PLModule, self).__init__()
        self.max_seq_len = max_seq_len
        self.pos_wt = pos_wt
        self.lr = lr
        self.model = MentionModel()
        self.l1_loss = torch.nn.L1Loss()
        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(self.pos_wt).to(self.device))
        self.save_hyperparameters()

    def forward(self, ids, mask):
        return self.model.forward(ids, mask)

    def _common_step(self, batch, mode: str):
        tok_id_list, mask, is_mention_list = batch
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
            on_step=True,
            on_epoch=False,
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
        correct = (pred_cls == true_cls)*mask
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
