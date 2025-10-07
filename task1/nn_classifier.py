import tensorflow as tf
from mnist_classifier_interface import MNISTClassifierInterface


class NNClassifier(MNISTClassifierInterface):
    """
    A simple feedforward neural network for MNIST classification.
    The input layer has 784 neurons (28*28 pixels flattened to a vector).
    Uses two dense layers. The first one has 128 neurons and ReLu activation function, which provides
    non-linearity.
    The second one has 10 neurons and Softmax function, which converts raw output, or
    logits into probabilities. 10 neurons correspond to 10 classes (digits from 0 to 9).

    Uses Adam optimizer with a learning rate of 0.001 and sparse categorical crossentropy as the loss function.
    Adam (Adaptive Moment Estimation) is a popular optimization algorithm that is super powerful.

    Sparse categorical crossentropy is used in multi-class classification, when the target labels
    are integers (not one-hot encoded).
    That's why while we're doing predictions, we use argmax to get the index of the highest probability
    as the predicted class label.

    Trains the model for 5 epochs with a batch size of 64.
    """

    def __init__(self):
        self.model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(shape=(28 * 28,)),  # Input layer of size 784 (28*28)
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

    def train(self, x_train, y_train):
        """
        Trains the model on the given training data, x and y.
        """
        self.model.fit(x_train, y_train, epochs=5, batch_size=64)

    def predict(self, x_test):
        """
        Predicts the labels for the given test data, x.
        """
        predictions = self.model.predict(x_test)
        return predictions.argmax(axis=1)
