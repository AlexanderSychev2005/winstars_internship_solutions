from abc import ABC, abstractmethod


class MNISTClassifierInterface(ABC):
    """
    Interface for MNIST different classifiers.

    Classifier should have methods:
    - train(x_train, y_train)
    - predict(x_test)
    """

    @abstractmethod
    def train(self, x_train, y_train):
        pass

    @abstractmethod
    def predict(self, x_test):
        pass
