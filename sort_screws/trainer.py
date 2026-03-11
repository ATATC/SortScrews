from typing import override

import torch
from mipcandy import Trainer, Params, AmbiguousShape, TrainerToolbox, convert_logits_to_ids
from torch import optim, nn

from sort_screws.network import EfficientNetNetwork


class EfficientNetTrainer(EfficientNetNetwork, Trainer):
    @override
    def build_optimizer(self, params: Params) -> optim.Optimizer:
        return optim.AdamW(params, lr=1e-3, weight_decay=1e-4)

    @override
    def build_scheduler(self, optimizer: optim.Optimizer, num_epochs: int) -> optim.lr_scheduler.LRScheduler:
        return optim.lr_scheduler.ConstantLR(optimizer, factor=1, total_iters=num_epochs * len(self._dataloader))

    @override
    def build_criterion(self) -> nn.Module:
        return nn.CrossEntropyLoss()

    @override
    def build_ema(self, model: nn.Module) -> nn.Module:
        return optim.swa_utils.AveragedModel(model)

    @override
    def _build_toolbox(self, num_epochs: int, example_shape: AmbiguousShape, compile_model: bool, ema: bool, *,
                       model: nn.Module | None = None) -> TrainerToolbox:
        if not model:
            model = self.load_model(example_shape, compile_model)
        optimizer = self.build_optimizer([p for p in model.parameters() if p.requires_grad])
        scheduler = self.build_scheduler(optimizer, num_epochs)
        criterion = self.build_criterion().to(self._device)
        return TrainerToolbox(model, optimizer, scheduler, criterion, self.build_ema(model) if ema else None)

    @override
    def backward(self, images: torch.Tensor, labels: torch.Tensor, toolbox: TrainerToolbox) -> tuple[float, dict[
        str, float]]:
        logits = toolbox.model(images)
        loss = toolbox.criterion(logits, labels)
        loss.backward()
        return loss.item(), {}

    @override
    def validate_case(self, idx: int, image: torch.Tensor, label: torch.Tensor, toolbox: TrainerToolbox) -> tuple[
        float, dict[str, float], torch.Tensor]:
        image, label = image.unsqueeze(0), label.unsqueeze(0)
        logits = toolbox.model(image)
        loss = toolbox.criterion(logits, label)
        return -loss.item(), {"correctness": float((convert_logits_to_ids(logits) == label).item())}, logits.squeeze(0)
