from typing import override

from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0, ResNet18_Weights, resnet18
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
            param.requires_grad = True if name.startswith("classifier") else self.is_backbone_trainable
        return model


class ResNetNetwork(WithNetwork):
    num_classes: int = 7
    is_backbone_trainable: bool = False
    @override
    def build_network(self, example_shape: AmbiguousShape) -> nn.Module:
        weights = ResNet18_Weights.DEFAULT
        model = resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, self.num_classes)
        # make backbone trainable or frozen
        for name, param in model.named_parameters():
            param.requires_grad = True if name.startswith("fc") else self.is_backbone_trainable
        return model
