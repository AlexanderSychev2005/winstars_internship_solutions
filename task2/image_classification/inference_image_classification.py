from pathlib import Path

import numpy as np
from tensorflow.keras.models import load_model
import cv2
import argparse

LABELS = [
    "cane",
    "cavallo",
    "elefante",
    "farfalla",
    "gallina",
    "gatto",
    "mucca",
    "pecora",
    "ragno",
    "scoiattolo",
]

TRANSLATIONS = {
    "cane": "dog",
    "cavallo": "horse",
    "elefante": "elephant",
    "farfalla": "butterfly",
    "gallina": "chicken",
    "gatto": "cat",
    "mucca": "cow",
    "pecora": "sheep",
    "scoiattolo": "squirrel",
    "ragno": "spider",
}


def preprocess_image(img_path, target_size):
    """
    Preprocess the image for model prediction. Provides resizing, normalization, and adding one more dimension, which is
    required by the model.

    :param img_path: Path to the input image
    :param target_size: Target size for resizing the image
    :return: Preprocessed image ready for model prediction
    """
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img / 255.0
    img = np.expand_dims(
        img, axis=0
    )  # Add one more dimension, (1, height, width, channels)
    return img


def predict_image(img_path, model_path, target_size=(256, 256)):
    """
    Predicts the class of an image using a trained EfficientNetB0 model. Chooses the class with the highest probability,
    and translates it to English. Also, returns the confidence score of the prediction.

    :param img_path: Path to the input image
    :param model_path: Path to the trained image classification model
    :param target_size: Target size for resizing the image
    :return: Predicted label (animal) and confidence score
    """
    model = load_model(model_path)
    img_array = preprocess_image(img_path, target_size)
    predictions = model.predict(img_array)

    predicted_index = np.argmax(predictions, axis=1)[0]
    predicted_label_translated = TRANSLATIONS[LABELS[predicted_index]]
    confidence = predictions[0][predicted_index]

    return predicted_label_translated, confidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Image classification inference script"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/final_model.h5",
        help="Path to the trained model",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default="examples/cat.png",
        help="Path to the input image",
    )
    parser.add_argument(
        "--target_size",
        type=int,
        nargs=2,
        default=(256, 256),
        help="Target image size (width height)",
    )

    args = parser.parse_args()
    # model_path = Path(args.model_path)
    predicted_label_translated, confidence = predict_image(
        args.image_path, args.model_path, args.target_size
    )
    print(
        f"Predicted label: {predicted_label_translated}, Confidence: {confidence:.4f}"
    )
