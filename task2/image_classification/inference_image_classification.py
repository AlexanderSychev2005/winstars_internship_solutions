import numpy as np
from tensorflow.keras.models import load_model
import cv2
from pathlib import Path

labels = [
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
translate = {
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

MODEL_PATH = "models/final_model.h5"
TARGET_SIZE = (256, 256)


def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, TARGET_SIZE)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img


def predict_image(model, img_path):
    img_array = preprocess_image(img_path)
    predictions = model.predict(img_array)

    predicted_index = np.argmax(predictions, axis=1)[0]
    predicted_label_translated = translate[labels[predicted_index]]
    confidence = predictions[0][predicted_index]

    return predicted_label_translated, confidence


if __name__ == "__main__":
    model = load_model(MODEL_PATH)
    print("Model loaded successfully.")

    test_image_path = Path("examples/belka.png")
    predicted_label_translated, confidence = predict_image(model, test_image_path)

    print(
        f"Predicted label: {predicted_label_translated}, Confidence: {confidence:.4f}"
    )
