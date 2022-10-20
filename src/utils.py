from functools import partial
import torch
from os.path import exists


def loss_fn(pred, label, fn):
    loss = 0
    lmda = 0.001
    for i in range(len(label)):
        for iw in range(len(label[i])):
            pred_oh = pred[i][iw]
            label_oh = label[i][iw]
            zero_class = label_oh[0] == 0
            loss_iw = fn(pred_oh, label_oh)
            if zero_class:
                loss = loss + lmda * loss_iw
            else:
                loss = loss + loss_iw

    return loss


def accuracy_fn(pred, label, mode="TRA"):
    pred = pred.detach()  # B , L , C
    label = label.detach()  # B, L
    correct = 0
    total = 0
    pred = torch.argmax(pred, dim=2)  # B, L
    mask = (label != 0) | True
    mask2 = label != 0
    correct = (label == pred) * mask
    print(f"{mode} all acc:  {correct.sum()}/{mask.sum()}")
    print(f"{mode} pro acc:  {(correct*mask2).sum()}/{mask2.sum()}")
    return correct.sum() / mask.sum()
    # for i in range(len(label)):
    #     for iw in range(len(label[i])):
    #         true_class = label[i][iw]
    #         pred_class = torch.argmax(pred[i][iw])
    #         if true_class == 0:
    #             continue
    #         if pred_class == true_class:
    #             correct += 1
    #         total += 1
    # return correct/total


class Trainer:
    def __init__(
        self,
        model_name,
        model,
        criterion,
        optimizer,
        scheduler,
        epochs,
        train_loader,
        val_loader,
        device,
        checkpoint_dir,
        early_stopping,
        log_periodicity,
        **kwargs,
    ):
        self.model_name = model_name
        self.model = model
        self.criterion = criterion
        # self.criterion = partial(loss_fn, fn=criterion)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epochs = epochs
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.early_stopping = early_stopping
        if self.checkpoint_dir[-1] != "/":
            self.checkpoint_dir += "/"
        self.log_periodicity = log_periodicity
        self.kwargs = kwargs
        self.sigmoid = torch.nn.Sigmoid()

    def train(self):
        print(f"Started Training of {self.model_name}")
        self.model = self.model.to(self.device)
        self.model.train()

        for epoch in range(self.epochs):
            running_loss = 0.0
            running_acc = 0.0

            # Train for an epoch
            for i, (ids, masks, labels) in enumerate(self.train_loader):
                ids = ids.to(self.device)
                masks = masks.to(self.device)
                labels = labels.to(self.device)

                # ids,masks are  B*seq_len
                # labels  B* (#classes)
                loss = 0
                out = self.model(ids, masks)  # B *L* (#classes)

                loss = self.criterion(out.view(-1, out.size(-1)), labels.view(-1))

                # Update the weights
                self.optimizer.zero_grad()
                running_loss += loss.detach()
                running_acc += accuracy_fn(out, labels)
                loss.backward()
                self.optimizer.step()

            if True:
                print(
                    "[%d, %d] loss: %.6f acc: %.4f"
                    % (
                        epoch + 1,
                        i + 1,
                        running_loss / len(self.train_loader),
                        running_acc / len(self.train_loader),
                    ),
                    flush=True,
                )
                running_loss = 0.0
                running_acc = 0.0

            # Calculate validation loss
            val_loss, accuracy = self.calculate_loss_accuracy(loader=self.val_loader)
            print(
                "Epoch: %d Validation loss: %.5f accuracy: %.5f"
                % (epoch + 1, val_loss, accuracy),
                flush=True,
            )

            # Take a scheduler step
            self.scheduler.step(val_loss)

            # Take a early stopping step
            self.early_stopping.step(val_loss)

            # Save a checkpoint according to strategy
            self.checkpoint(epoch)

            # Check early stopping to finish training
            if self.early_stopping.stop_training:
                print("Early Stopping the training")
                break

            # check external instructions to stop training
            if exists("STOP"):
                print("External instruction to stop the training")
                break

        print("Finished Training")
        self.save_model(self.model_name + "final")

    def calculate_loss_accuracy(self, loader):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        total_acc = 0

        # Test validation data
        with torch.no_grad():
            for i, (ids, masks, labels) in enumerate(loader):
                ids = ids.to(self.device)
                masks = masks.to(self.device)
                labels = labels.to(self.device)

                # ids,masks are  B*seq_len
                # labels  B* (#classes)
                loss = 0
                pred_classes = self.model(ids, masks)  # B * (#classes)
                loss = self.criterion(pred_classes, labels)

                # Applying sigmoid
                pred_classes = self.sigmoid(pred_classes)

                # Accumulating loss, accuracy
                acc = accuracy_fn(pred_classes, labels, "val")
                total_loss += loss
                total_acc += acc

        self.model.train()
        return total_loss / len(loader), total_acc / len(loader)

    def save_model(self, file_name):
        print(f"Saving Model in {self.checkpoint_dir + file_name}")
        torch.save(self.model.state_dict(), f"{self.checkpoint_dir + file_name}.pt")

    def evaluate(self, name, loader):
        loss, accuracy = self.calculate_loss_accuracy(loader)
        print(f"{name} loss = {loss} \t {name} accuracy = {accuracy}")

    def checkpoint(self, epoch):
        save_checkpoint = False
        checkpoint_name = ""
        if self.kwargs["checkpoint_strategy"] == "periodic" and epoch % self.kwargs[
            "checkpoint_periodicity"
        ] == (self.kwargs["checkpoint_periodicity"] - 1):
            save_checkpoint = True
            checkpoint_name = f"checkpoint_{epoch}"
        elif (
            self.kwargs["checkpoint_strategy"] == "best"
            and self.early_stopping.current_count == 0
        ):
            save_checkpoint = True
            checkpoint_name = "checkpoint_best"
        elif self.kwargs["checkpoint_strategy"] == "both":
            if self.early_stopping.current_count == 0:
                save_checkpoint = True
                checkpoint_name = "checkpoint_best"
            elif epoch % self.kwargs["checkpoint_periodicity"] == (
                self.kwargs["checkpoint_periodicity"] - 1
            ):
                save_checkpoint = True
                checkpoint_name = f"checkpoint_{epoch}"

        if save_checkpoint:
            self.save_model(self.model_name + checkpoint_name)


class EarlyStopping:
    def __init__(self, patience):
        self.patience = patience
        self.min_loss = 1e5
        self.current_count = 0
        self.stop_training = False

    def step(self, val_loss):
        if val_loss < self.min_loss:
            self.current_count = 0
            self.min_loss = val_loss
        else:
            self.current_count += 1

        if self.current_count >= self.patience:
            self.stop_training = True
