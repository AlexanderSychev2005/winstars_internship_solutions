from sklearn.ensemble import RandomForestClassifier
from mnist_classifier_interface import MNISTClassifierInterface


class RandomForestClassifierModel(MNISTClassifierInterface):
    """
    Random Forest Classifier for MNIST classification.
    Uses 300 trees in the forest and a fixed random state for reproducibility (connected to
    pseudo-random values).
    While we're predicting, it directly returns the predicted class labels.
    """
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=300, random_state=42)

    def train(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def predict(self, x_test):
        predictions = self.model.predict(x_test)
        return predictions
