import torch
from torch import nn
from model import TorchGRUIntent, TorchLSTMIntent, TransformerIntent
from utils import Trainer, EarlyStopping
from dataset import get_dataloaders, mLangDataset
import argparse

parser = argparse.ArgumentParser(description="Get configurations to train")
parser.add_argument("--cpu_cores", default=10, type=int)
parser.add_argument("--data", default="", type=str)
parser.add_argument("--model_type", default="TGRU", type=str)
parser.add_argument("--model", default="", type=str)
parser.add_argument("--mode", default="train", type=str)
CONFIG = parser.parse_args()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", device)

checkpoint_dir = "checkpoints/"

# Hyperparameters
batch_size = 128
learning_rate = 0.001
epochs = int(2e7)
cpu_cores = CONFIG.cpu_cores
print(f"Using {cpu_cores} CPU cores")

train_loader, val_loader, test_loader, max_seq_len = get_dataloaders(
    mLangDataset, {"batch_size": batch_size, "test_split": 0.00, "val_split": 0.3}
)
print(f"Train DS len: {len(train_loader)}")
print(f"Val DS len: {len(val_loader)}")
print(f"Test DS len: {len(test_loader)}")
print(f"Max sequence len: {max_seq_len}")
# Getting the model
if CONFIG.model_type == "TGRU":
    model = TorchGRUIntent(hidden_size=300, seq_len=max_seq_len)
elif CONFIG.model_type == "TLSTM":
    model = TorchLSTMIntent(hidden_size=300, seq_len=max_seq_len)
elif CONFIG.model_type == "Transformer":
    model = TransformerIntent(seq_len=max_seq_len)
else:
    print("Unidentified model type")
    exit(1)

if CONFIG.model == "":
    print("Training new model ", type(model))
else:
    print("Using model from", CONFIG.model)
    model.load_state_dict(torch.load(CONFIG.model))
    model = model.to(device)


# Optimizer and Criterion
criterion = nn.CrossEntropyLoss(ignore_index=0)
crit_wt = torch.ones(max_seq_len).to(device)
crit_wt[0] = 0.0001
criterion = nn.CrossEntropyLoss(weight=crit_wt)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer=optimizer, mode="min", factor=0.5, patience=400, verbose=True
)


# Early Stopping
early_stopping = EarlyStopping(patience=99999)

# Train the model
trainer = Trainer(
    model_name="BERT_TGRU_2b_mha_snips",
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    epochs=epochs,
    train_loader=train_loader,
    val_loader=val_loader,
    device=device,
    checkpoint_dir=checkpoint_dir,
    early_stopping=early_stopping,
    log_periodicity=1,
    checkpoint_strategy="both",
    checkpoint_periodicity=10,
)

if CONFIG.mode == "train":
    trainer.train()

# Test
trainer.evaluate(name="Val", loader=val_loader)
# trainer.evaluate(name="Test", loader=test_loader)
