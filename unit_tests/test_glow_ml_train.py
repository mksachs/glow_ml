import pytest

from glow_ml_train import train
from glow_ml_train.exceptions import AlgorithmNotImplemented


def test_predict_accident():
    with pytest.raises(AlgorithmNotImplemented):
        train.predict_accident(None, algorithm='sf')

    with pytest.raises(ValueError):
        train.predict_accident(None)
