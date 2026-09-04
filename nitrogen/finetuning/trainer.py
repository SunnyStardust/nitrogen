from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from nitrogen.cfg import CkptConfig

from .checkpoint import export_inference_checkpoint, save_training_checkpoint


@dataclass
class FineTuningConfig:
    output_dir: Path
    learning_rate: float = 1e-4
    max_steps: int = 100
    gradient_clip_norm: float = 1.0
    save_every: int = 100


def move_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict:
    return {key: value.to(device) for key, value in batch.items()}


class FineTuningTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        config: FineTuningConfig,
        ckpt_config: CkptConfig,
        game_mapping: dict | None,
        device: torch.device,
    ):
        self.model = model
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.config = config
        self.ckpt_config = ckpt_config
        self.game_mapping = game_mapping
        self.device = device
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if not parameters:
            raise ValueError("No trainable model parameters")
        self.optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate)
        self.step = 0
        config.output_dir.mkdir(parents=True, exist_ok=True)

    def train(self) -> list[float]:
        losses = []
        iterator = iter(self.train_loader)
        self.model.train()
        while self.step < self.config.max_steps:
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(self.train_loader)
                batch = next(iterator)
            batch = move_to_device(batch, self.device)
            self.optimizer.zero_grad(set_to_none=True)
            loss = self.model(batch)["loss"]
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss at step {self.step}: {loss}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in self.model.parameters() if parameter.requires_grad],
                self.config.gradient_clip_norm,
            )
            self.optimizer.step()
            self.step += 1
            losses.append(float(loss.detach()))
            print(f"step={self.step} loss={losses[-1]:.6f}")
            if self.step % self.config.save_every == 0:
                self.save()
        return losses

    @torch.no_grad()
    def validate(self) -> float:
        self.model.eval()
        losses = []
        for batch in self.validation_loader:
            loss = self.model(move_to_device(batch, self.device))["loss"]
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite validation loss")
            losses.append(float(loss))
        self.model.train()
        if not losses:
            raise ValueError("Validation loader contains no batches")
        return sum(losses) / len(losses)

    def save(self) -> Path:
        path = self.config.output_dir / "training.pt"
        save_training_checkpoint(
            path,
            self.model,
            self.optimizer,
            self.step,
            self.ckpt_config,
            self.game_mapping,
        )
        return path

    def export(self) -> Path:
        path = self.config.output_dir / "inference.pt"
        export_inference_checkpoint(
            path,
            self.model,
            self.ckpt_config,
            self.game_mapping,
        )
        return path
