from image_classification.inference_image_classification import predict_image
from ner_classification.infer_bert import predict_animal
import argparse


def is_predictions_equal(image, text, image_target_size):
    image_model_path = "./image_classification/models/final_model.h5"
    text_model_path = "./ner_classification/models/ner_model_bert"
    image_prediction, _ = predict_image(image, image_model_path, image_target_size)
    text_prediction = predict_animal(text, text_model_path)
    return image_prediction == text_prediction


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pipeline to compare predictions from the image classification model \
     and the NER model"
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default="image_classification/examples/cat.png",
        help="Path to the image file",
    )
    parser.add_argument(
        "--text_input",
        type=str,
        default="I love that cat!",
        help="Input text containing an animal entity",
    )
    parser.add_argument(
        "--image_target_size",
        type=tuple,
        default=(256, 256),
        help="Target size for image resizing (image will be resized to target_size)",
    )
    args = parser.parse_args()

    if is_predictions_equal(args.image_path, args.text_input, args.image_target_size):
        print("The predictions from both models are the same.")
    else:
        print("The predictions from both models are different.")
