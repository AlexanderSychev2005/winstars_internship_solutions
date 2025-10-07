from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
import kagglehub
import tensorflow as tf
from pathlib import Path
import argparse


def build_model(num_classes, input_shape):
    """
    Builds and returns an EfficientNetB0 model with custom-made top layers for classification.
    The EfficientNet architecture is represented by a set of ready-to-use models. The choice depends on the required
    accuracy, available training resources, and input image resolution. Models are labelled from B0 (simplest)
    to B7 (most powerful).
    I'll use EfficientNet-B0 architecture with the weights from ImageNet dataset. EfficientNet-B0 is trained on the
    ImageNet dataset and can classify images into 1000 object categories. The point is to re-use the weights from the
    pre-trained models downloading and using directly or integrating into a new model that we'll do.

    The include_top parameter is set to False, so the network does not include he top layer/ output layer.
    We have a possibility to add our own output layer depending on our needs. (10 values)

    :param num_classes: Number of classes for classification (10 for Animals10 dataset)
    :param input_shape: Shape of the input images (height, width, channels)
    :return: Ready for training model
    """
    print("Input shape:", input_shape)
    base_model = EfficientNetB0(
        include_top=False, weights="imagenet", input_shape=input_shape
    )
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    x = Dense(num_classes, activation="softmax")(x)
    model_eff = Model(inputs=base_model.input, outputs=x)
    return model_eff


def get_dataset(data_path, batch_size, target_size):
    """
    Loads the dataset from the dataset directory, and splits it into training and validation sets.
    The images there are resized to the target size, and normalized to the range [0, 1].
    One hot encoding is used to represent the labels.
    The dataset is returned as TensorFlow Dataset objects for both training and validation, which are efficient because
    we use generators to load the data in batches during training, so that we don't load the entire dataset into memory at once.
    80% of the data is used for training, and 20% for validation.
    10 classes are used, corresponding to the 10 animal categories in the Animals10 dataset.

    :param data_path: Path to the dataset directory
    :param batch_size: Batch size for training
    :param target_size: Target size for resizing the images (height, width)
    :return: Training and validation datasets
    """

    train_dataset = tf.keras.utils.image_dataset_from_directory(
        data_path,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=target_size,
        batch_size=batch_size,
        color_mode="rgb",
    )
    val_dataset = tf.keras.utils.image_dataset_from_directory(
        data_path,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=target_size,
        batch_size=batch_size,
        color_mode="rgb",
    )

    train_dataset = train_dataset.map(
        lambda x, y: (x / 255.0, tf.one_hot(y, depth=10))
    )  # Normalizing and one-hot encoding
    val_dataset = val_dataset.map(lambda x, y: (x / 255.0, tf.one_hot(y, depth=10)))

    # train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    # val_dataset = val_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    for images, labels in train_dataset.take(1):
        print(images.shape, labels.shape)

    test_image = next(iter(train_dataset.take(1)))[0][0].numpy()
    print("Test image shape:", test_image.shape)

    return train_dataset, val_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Training EfficientNetB0 model on Animals10 dataset"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to the dataset (default: download from Kaggle)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--target_size",
        type=int,
        default=256,
        help="Target image size (images will be resized to target_size x target_size)",
    )
    parser.add_argument(
        "--epochs", type=int, default=12, help="Number of training epochs"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Learning rate for optimizer"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/final_model.h5",
        help="Path to save final model",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="models/effnet_checkpoint.h5",
        help="Path to save checkpoint",
    )
    parser.add_argument(
        "--log_dir", type=str, default="logs", help="Directory for TensorBoard logs"
    )

    args = parser.parse_args()

    if args.data_path:
        data_path = Path(args.data_path)
    else:
        path = kagglehub.dataset_download("alessiocorrado99/animals10")
        data_path = Path(f"{path}/raw-img")

    train_dataset, val_dataset = get_dataset(
        data_path, args.batch_size, (args.target_size, args.target_size)
    )
    input_shape = (args.target_size, args.target_size, 3)

    model = build_model(num_classes=10, input_shape=input_shape)

    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    tensorboard = TensorBoard(log_dir="logs")
    checkpoint = ModelCheckpoint(
        args.checkpoint_path,
        monitor="val_accuracy",
        save_best_only=True,
        mode="auto",
        verbose=1,
    )
    reduce_lr = ReduceLROnPlateau(
        monitor="val_accuracy",
        factor=0.3,
        patience=2,
        min_delta=0.001,
        mode="auto",
        verbose=1,
    )

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=args.epochs,
        verbose=1,
        callbacks=[checkpoint, tensorboard, reduce_lr],
    )

    model_dir = Path(args.model_path).parent
    model_dir.mkdir(parents=True, exist_ok=True)

    model.save(args.model_path)
