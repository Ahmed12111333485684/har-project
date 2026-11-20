from .cnn_bilstm import CNNBiLSTM
from .transformer import HARTransformer
from .resnet1d import ResNet1D
from .baseline import build_sklearn

DL_MODELS = {"cnn_bilstm": CNNBiLSTM, "transformer": HARTransformer, "resnet1d": ResNet1D}
SKLEARN_MODELS = ("rf", "svm")


def build_model(name, in_channels, n_classes, **params):
    return DL_MODELS[name](in_channels, n_classes, **params)
