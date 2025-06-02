from __future__ import annotations

from typing import Any, Dict, Optional, Type

import pytorch_lightning as pl
import torch


class LightningModelAdapter(pl.LightningModule):
    """
    A PyTorch Lightning adapter that wraps a torch.nn.Module.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn: Any,
        optimizer_cls: Type[torch.optim.Optimizer],
        optimizer_params: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer_cls = optimizer_cls
        self.optimizer_params = optimizer_params or {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the underlying model."""
        return self.model(x)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """
        Implements a single training step.

        Expects batch to be a tuple of (inputs, targets).
        """
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """
        Implements a single validation step.

        Expects batch to be a tuple of (inputs, targets).
        """
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Instantiate the optimizer using the provided class and parameters."""
        return self.optimizer_cls(self.model.parameters(), **self.optimizer_params)

    # Optional: convenience methods for saving/loading weights
    def save(self, filepath: str) -> None:
        """Persist the model's state_dict to a file."""
        torch.save(self.model.state_dict(), filepath)

    @classmethod
    def load(
        cls,
        filepath: str,
        model_architecture: torch.nn.Module,
        loss_fn: Any,
        optimizer_cls: Type[torch.optim.Optimizer],
        optimizer_params: Optional[Dict[str, Any]] = None,
    ) -> LightningModelAdapter:
        """
        Load a persisted state_dict into the provided architecture
        and wrap it in a LightningModelAdapter.
        """
        state_dict = torch.load(filepath, map_location="cpu")
        model_architecture.load_state_dict(state_dict)
        return cls(model_architecture, loss_fn, optimizer_cls, optimizer_params)
