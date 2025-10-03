import tensorflow as tf
from mnist_classifier_interface import MNISTClassifierInterface


class CNNClassifier(MNISTClassifierInterface):
    """
    A Convolutional Neural Network (CNN) for MNIST classification.
    The input layer has shape (28, 28, 1) - 28x28 pixels with 1 extra dimension -  color channel
    (grayscale).
    Uses two convolutional layers followed by max-pooling layers to reduce spatial dimensions.
    The first convolutional layer has 24 filters, the second one has 36 filters.
    Both use ReLu activation function, which provides non-linearity.
    After the convolutional layers, the output is flattened and passed through a dense layer with
    128 neurons and ReLu activation.
    Finally, the output layer has 10 neurons with Softmax activation, which converts raw output, or
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
            tf.keras.layers.InputLayer(shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(filters=24, kernel_size=(3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(filters=36, kernel_size=(3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),

            tf.keras.layers.Dense(10, activation='softmax')
        ])
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

    def train(self, x_train, y_train):
        self.model.fit(x_train, y_train, epochs=5, batch_size=64)

    def predict(self, x_test):
        predictions = self.model.predict(x_test)
        return predictions.argmax(axis=1)
