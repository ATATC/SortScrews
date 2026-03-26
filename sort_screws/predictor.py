from mipcandy import Predictor

from sort_screws.network import EfficientNetNetwork, ResNetNetwork, SwinV2Network


class EfficientNetPredictor(EfficientNetNetwork, Predictor):
    pass


class ResNetPredictor(ResNetNetwork, Predictor):
    pass


class SwinV2Predictor(SwinV2Network, Predictor):
    pass
