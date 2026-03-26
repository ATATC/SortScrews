from typing import override

from mipcandy import WithNetwork, AmbiguousShape
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0, ResNet18_Weights, resnet18, \
    Swin_V2_T_Weights, swin_v2_t, ConvNeXt_Small_Weights, convnext_small


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

    @staticmethod
    def get_transforms() -> nn.Module:
        return EfficientNet_B0_Weights.DEFAULT.transforms()


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

    @staticmethod
    def get_transforms() -> nn.Module:
        return ResNet18_Weights.DEFAULT.transforms()


class SwinV2Network(WithNetwork):
    num_classes: int = 7
    is_backbone_trainable: bool = False

    @override
    def build_network(self, example_shape: AmbiguousShape) -> nn.Module:
        weights = Swin_V2_T_Weights.DEFAULT
        model = swin_v2_t(weights=weights)
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, self.num_classes)
        for name, param in model.named_parameters():
            if name.startswith("head"):
                param.requires_grad = True
            else:
                param.requires_grad = self.is_backbone_trainable
        return model

    @staticmethod
    def get_transforms() -> nn.Module:
        return Swin_V2_T_Weights.DEFAULT.transforms()


class ConvNeXtNetwork(WithNetwork):
    num_classes: int = 7
    is_backbone_trainable: bool = True

    @override
    def build_network(self, example_shape: AmbiguousShape) -> nn.Module:
        weights = ConvNeXt_Small_Weights.DEFAULT
        model = convnext_small(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Sequential(nn.LayerNorm(in_features), nn.Flatten(),
                                             nn.Linear(in_features, self.num_classes))
        for name, param in model.named_parameters():
            param.requires_grad = (
                True if name.startswith("classifier") else self.is_backbone_trainable
            )
        return model

    @staticmethod
    def get_transforms() -> nn.Module:
        return ConvNeXt_Small_Weights.DEFAULT.transforms()
