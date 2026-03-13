from mipcandy import Predictor

from sort_screws.network import EfficientNetNetwork, ResNetNetwork


class EfficientNetPredictor(EfficientNetNetwork, Predictor):
    pass


class ResNetPredictor(ResNetNetwork, Predictor):
    pass
