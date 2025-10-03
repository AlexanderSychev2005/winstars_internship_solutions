from cnn_classifier import CNNClassifier
from nn_classifier import NNClassifier
from random_forest_classifier import RandomForestClassifierModel


class MnistClassifier:
    """
    An interface to choose between different algorithms for MNIST classification. Hides all the functionality.
    """
    def __init__(self, algorithm):
        self.algorithm = algorithm
        if algorithm == 'rf':
            self.model = RandomForestClassifierModel()
        elif algorithm == 'nn':
            self.model = NNClassifier()
        elif algorithm == 'cnn':
            self.model = CNNClassifier()
        else:
            raise ValueError("There is no such algorithm there. Please choose 'rf', 'nn' or 'cnn'.")

    def train(self, x_train, y_train):
        """
        Trains the model on the given training data.
        Reshapes the data according to the algorithm used. If we use CNN, we need to add an extra dimension
        to the data. Otherwise, we flatten the images to 1D vectors (28 * 28) = (784).
        """
        if self.algorithm == 'cnn':
            x_train = x_train.reshape(-1, 28, 28, 1)
            return self.model.train(x_train, y_train)

        x_train = x_train.reshape(-1, 28 * 28)
        return self.model.train(x_train, y_train)

    def predict(self, x_test):
        """
        Predicts the labels for the given test data.
        As in the train method, we need to reshape the data according to the algorithm used.
        """
        if self.algorithm == 'cnn':
            x_test = x_test.reshape(-1, 28, 28, 1)
            return self.model.predict(x_test)

        x_test = x_test.reshape(-1, 28 * 28)
        return self.model.predict(x_test)
