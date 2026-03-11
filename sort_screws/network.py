from typing import override

from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
from mipcandy import WithNetwork, AmbiguousShape


class EfficientNetNetwork(WithNetwork):
    num_classes: int = 7
    is_backbone_trainable: bool = False
    @override
    def build_network(self, example_shape: AmbiguousShape) -> nn.Module:
        weights = EfficientNet_B0_Weights.DEFAULT
        model = efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, self.num_classes)
        # make backbone trainable or frozen
        for name, param in model.named_parameters():
            if not name.startswith("classifier"):
                param.requires_grad = self.is_backbone_trainable
        for name, param in model.named_parameters():
            if name.startswith("classifier"):
                param.requires_grad = True
        return model
