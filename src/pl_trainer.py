import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.plugins.io import AsyncCheckpointIO
from dataset import get_dataloaders, MentionDataset, PairScoreDataset, get_mention_ratio
from pl_module import PLModuleMention, PLModulePairScore
import argparse
import configs


def train_mention():
    checkpoint_callback = ModelCheckpoint(
        dirpath="../logs/checkpoints",
        filename="checkpoint_{epoch:02d}_{train_loss:.4f}",
        save_top_k=50,
        monitor="train_loss",
        every_n_epochs=1,
        save_last=True,
        save_on_train_epoch_end=True,
    )
    lr_monitor_callback = LearningRateMonitor(logging_interval="epoch")
    async_ckpt_io = AsyncCheckpointIO()
    trainer = pl.Trainer(
        accelerator="gpu",
        benchmark=True,
        precision=32,
        callbacks=[checkpoint_callback, lr_monitor_callback],
        plugins=[async_ckpt_io],
        max_epochs=300,
        log_every_n_steps=1,
        default_root_dir="../logs"
        # fast_dev_run=True,
    )
    ds = MentionDataset(include_lang=configs.include_langs)
    train_loader, val_loader, test_loader = get_dataloaders(
        ds,
        {"val_split": 0.2, "test_split": 0, "batch_size": configs.mention_batch_size},
    )
    pl_module_mention = PLModuleMention(
        ds.MAX_SEQ_LEN,
        # pos_wt=200,
        pos_wt=get_mention_ratio(ds),
    )
    trainer.fit(pl_module_mention, train_loader, val_loader)


def train_pair_score(get_stats=False):
    checkpoint_callback = ModelCheckpoint(
        dirpath="../logs/checkpoints",
        filename="checkpoint_{epoch:02d}_{train_loss:.4f}",
        save_top_k=50,
        monitor="train_loss",
        every_n_epochs=1,
        save_last=True,
        save_on_train_epoch_end=True,
    )
    lr_monitor_callback = LearningRateMonitor(logging_interval="epoch")
    async_ckpt_io = AsyncCheckpointIO()
    trainer = pl.Trainer(
        accelerator="gpu",
        benchmark=True,
        precision=32,
        callbacks=[checkpoint_callback, lr_monitor_callback],
        plugins=[async_ckpt_io],
        max_epochs=300,
        log_every_n_steps=1,
        default_root_dir="../logs"
        # fast_dev_run=True,
    )
    ds = PairScoreDataset(
        mention_wt_path=configs.mention_model_path,
        get_stats=get_stats,
        include_lang=configs.include_langs,
    )
    train_loader, val_loader, test_loader = get_dataloaders(
        ds,
        {"val_split": 0.2, "test_split": 0, "batch_size": 2048 * 128},
    )
    print(len(train_loader))
    pl_module = PLModulePairScore("", ds.MAX_SEQ_LEN, pos_wt=111)
    trainer.fit(pl_module, train_loader, val_loader)


if __name__ == "__main__":
    # set argparser to train which model
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="mention")
    parser.add_argument("--stats", action="store_true")
    args = parser.parse_args()
    if args.model == "mention":
        train_mention()
    else:
        train_pair_score(args.stats)
