import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils as utils
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
from loguru import logger
import os
from tqdm import tqdm

# from transformers import Trainer


# class ContrastiveTrainer(Trainer):
#     def __init__(self, loss, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         # self.add_callback(EarlyStoppingCallback(early_stopping_patience=2))
#         self.loss = loss
#         self._signature_columns = ["rumor", "evidence", "negatives"]
#         self.can_return_loss = True  # Whether the model can return a loss

#     def compute_loss(self, model, inputs, return_outputs=False):
#         rumor = inputs.pop("rumor")
#         evidence = inputs.pop("evidence")
#         negatives = inputs.pop("negatives")

#         rumor_outputs = model(**rumor)["last_hidden_state"]
#         evidence_outputs = model(**evidence)["last_hidden_state"]
#         negative_outputs = model(**negatives)["last_hidden_state"]

#         # rumor_outputs = model(**rumor)["pooler_output"]
#         # evidence_outputs = model(**evidence)["pooler_output"]
#         # negative_outputs = model(**negatives)["pooler_output"]

#         loss = self.loss(rumor_outputs, evidence_outputs, negative_outputs)
#         outputs = {
#             "rumor": rumor_outputs,
#             "evidence": evidence_outputs,
#             "negatives": negative_outputs,
#         }
#         return (loss, outputs) if return_outputs else loss


# class ClassifierTrainer(Trainer):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self._signature_columns = ["rumor", "evidence", "label"]


class CustomTrainer:
    def __init__(
        self,
        model,
        train_dataset,
        val_dataset=None,
        batch_size=32,
        lr=0.001,
        max_grad_norm=1.0,
        device="cuda",
        distributed=False,
        patience=5,
        log_dir="logs",
        data_collator=None,
        save_model=True,
    ):
        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.lr = lr
        self.device = device
        self.max_grad_norm = max_grad_norm
        self.distributed = distributed
        self.patience = patience
        self.log_dir = log_dir
        self.data_collator = data_collator
        self.to_save_model = save_model

        # Initialize distributed training if specified
        if self.distributed:
            self.init_distributed()

    def init_distributed(self):
        dist.init_process_group(backend="nccl")
        self.model = DDP(self.model)

    def train(self, epochs):
        if self.distributed:
            parameters = self.model.module.parameters()
        else:
            parameters = self.model.parameters()

        # for n, p in self.model.named_parameters():
        #     if p.requires_grad:
        #         print(n)

        optimizer = optim.Adam(parameters, lr=self.lr, weight_decay=0.01)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

        if self.distributed:
            train_sampler = DistributedSampler(self.train_dataset)
        else:
            train_sampler = None

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=(train_sampler is None),
            # shuffle=False,
            sampler=train_sampler,
            collate_fn=self.data_collator,
        )

        # TensorBoard logging
        writer = SummaryWriter(log_dir=self.log_dir)

        logger.add(os.path.join(self.log_dir, "train.log"), rotation="500 MB")

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0

            with tqdm(
                total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}", unit="batch"
            ) as pbar:
                for i, inputs in enumerate(train_loader):
                    inputs = self.to_device(inputs, self.device)
                    optimizer.zero_grad()
                    loss = self.compute_loss(self.model, inputs)
                    loss.backward()
                    utils.clip_grad_norm_(
                        parameters, self.max_grad_norm
                    )  # Gradient clipping
                    optimizer.step()
                    running_loss += loss.item() * self.batch_size

                    pbar.update(1)
                    pbar.set_postfix(
                        {
                            "loss": running_loss / len(train_loader.dataset),
                            "lr": optimizer.param_groups[0]["lr"],
                        }
                    )

            epoch_loss = running_loss / len(self.train_dataset)
            # print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

            # Log to TensorBoard
            writer.add_scalar("Loss/train", epoch_loss, epoch)

            # Log to file
            logger.info(
                f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Lr: {optimizer.param_groups[0]['lr']}"
            )

            if self.val_dataset:
                val_loss = self.validate()
                # print(f"Validation Loss: {val_loss:.4f}")

                # Log to TensorBoard
                writer.add_scalar("Loss/val", val_loss, epoch)

                # Log to file
                logger.info(
                    f"Epoch {epoch+1}/{epochs}, Validation Loss: {val_loss:.4f}"
                )

                # Check for early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= self.patience:
                    print(
                        f"Validation loss hasn't improved for {self.patience} epochs. Early stopping..."
                    )
                    break
            # Update learning rate
            scheduler.step()

        # Close TensorBoard writer
        writer.close()
        if self.to_save_model:
            os.makedirs(os.path.join(self.log_dir, "checkpoints"), exist_ok=True)
            self.save_model(os.path.join(self.log_dir, "checkpoints"))

    def validate(self):
        if self.distributed:
            val_sampler = DistributedSampler(self.val_dataset)
        else:
            val_sampler = None

        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=val_sampler,
            collate_fn=self.data_collator,
        )

        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            with tqdm(
                total=len(val_loader), desc=f"Validation: ", unit="batch"
            ) as pbar:
                for inputs in val_loader:
                    inputs = self.to_device(inputs, self.device)
                    loss, outputs = self.compute_loss(
                        self.model, inputs, return_outputs=True
                    )
                    val_loss += loss.item() * self.batch_size
                    pbar.update(1)

        # accuracy = correct / total
        return val_loss / len(self.val_dataset)

    def to_device(self, data, device):
        if isinstance(data, torch.Tensor):
            return data.to(device)
        elif isinstance(data, dict):
            return {key: self.to_device(value, device) for key, value in data.items()}
        elif isinstance(data, (list, tuple)):
            return [self.to_device(item, device) for item in data]
        else:
            return data

    def compute_loss(self, model, inputs, return_outputs=False):
        raise NotImplementedError

    def save_model(self, path):
        if self.distributed:
            model = self.model.module
        else:
            model = self.model
        model.save_pretrained(path)


class CustomContrastiveTrainer(CustomTrainer):
    def __init__(self, loss, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss = loss

    def compute_loss(self, model, inputs, return_outputs=False):
        rumor = inputs.pop("rumor")
        evidence = inputs.pop("evidence")
        negatives = inputs.pop("negatives")

        rumor_outputs = model(**rumor)["last_hidden_state"]
        evidence_outputs = model(**evidence)["last_hidden_state"]
        negative_outputs = model(**negatives)["last_hidden_state"]

        loss = self.loss(rumor_outputs, evidence_outputs, negative_outputs)
        outputs = {
            "rumor": rumor_outputs,
            "evidence": evidence_outputs,
            "negatives": negative_outputs,
        }
        return (loss, outputs) if return_outputs else loss


class CustomClassifierTrainer(CustomTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss


class CustomJointTrainer(CustomTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        loss, outputs = model(inputs, return_outputs=True)
        return (loss, outputs) if return_outputs else loss

    def save_model(self, path):
        if self.distributed:
            model = self.model.module
        else:
            model = self.model
        model.save_model(path)
