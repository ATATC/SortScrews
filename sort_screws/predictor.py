from mipcandy import Predictor

from sort_screws.network import EfficientNetNetwork, ResNetNetwork, SwinV2Network, ConvNeXtNetwork


class EfficientNetPredictor(EfficientNetNetwork, Predictor):
    pass


class ResNetPredictor(ResNetNetwork, Predictor):
    pass


class SwinV2Predictor(SwinV2Network, Predictor):
    pass


class ConvNeXtPredictor(ConvNeXtNetwork, Predictor):
    pass
