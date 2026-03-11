from typing import override

import torch
from mipcandy import Trainer, Params, AmbiguousShape, TrainerToolbox, convert_logits_to_ids, save_image
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
        loss = toolbox.criterion(logits.softmax(1), labels)
        loss.backward()
        return loss.item(), {}

    @override
    def validate_case(self, idx: int, image: torch.Tensor, label: torch.Tensor, toolbox: TrainerToolbox) -> tuple[
        float, dict[str, float], torch.Tensor]:
        image, label = image.unsqueeze(0), label.unsqueeze(0)
        logits = toolbox.model(image)
        loss = toolbox.criterion(logits.softmax(1), label)
        pred_ids = convert_logits_to_ids(logits)
        metrics = {"correctness": float((pred_ids == label).item())}
        for class_id in range(self.num_classes):
            metrics[f"label cls {class_id}"] = 1 if class_id == label.item() else 0
            metrics[f"pred cls {class_id}"] = 1 if class_id == pred_ids.item() else 0
        return -loss.item(), metrics, logits.squeeze(0)

    @override
    def save_preview(self, image: torch.Tensor, label: torch.Tensor, output: torch.Tensor, *,
                     quality: float = .75) -> None:
        save_image(image, f"{self.experiment_folder()}/input.png")
        label = str(label.item())
        with open(f"{self.experiment_folder()}/label.txt", "w") as f:
            f.write(label)
        output = str(convert_logits_to_ids(output, channel_dim=0).item())
        with open(f"{self.experiment_folder()}/output.txt", "w") as f:
            f.write(output)
