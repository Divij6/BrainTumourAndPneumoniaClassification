from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
TRAINING_DIR = DATA_DIR / "brain_tumor_training"
TESTING_DIR = DATA_DIR / "brain_tumor_testing"
MODEL_PATH = BASE_DIR / "brain_tumor_model.h5"

IMAGE_SIZE = 150
LABELS = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]


def load_images_from(folder: Path, labels: list[str]) -> tuple[list[np.ndarray], list[str]]:
    images = []
    targets = []

    for label in labels:
        label_dir = folder / label
        if not label_dir.exists():
            raise FileNotFoundError(f"Missing dataset folder: {label_dir}")

        for image_path in label_dir.iterdir():
            if not image_path.is_file():
                continue

            img = cv2.imread(str(image_path))
            if img is None:
                print(f"Skipping unreadable image: {image_path}")
                continue

            img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
            images.append(img)
            targets.append(label)

    return images, targets


def build_model() -> Sequential:
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(150, 150, 3)))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(0.3))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(Dropout(0.3))
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(0.3))
    model.add(Conv2D(128, (3, 3), activation="relu"))
    model.add(Conv2D(128, (3, 3), activation="relu"))
    model.add(Conv2D(128, (3, 3), activation="relu"))
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(0.3))
    model.add(Conv2D(128, (3, 3), activation="relu"))
    model.add(Conv2D(256, (3, 3), activation="relu"))
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(512, activation="relu"))
    model.add(Dense(512, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(len(LABELS), activation="softmax"))
    return model


def main() -> None:
    train_images, train_labels = load_images_from(TRAINING_DIR, LABELS)
    test_images, test_labels = load_images_from(TESTING_DIR, LABELS)

    images = np.array(train_images + test_images)
    labels = np.array(train_labels + test_labels)
    images, labels = shuffle(images, labels, random_state=101)

    x_train, _, y_train, _ = train_test_split(
        images,
        labels,
        test_size=0.1,
        random_state=101,
    )

    y_train = tf.keras.utils.to_categorical([LABELS.index(label) for label in y_train])

    model = build_model()
    model.summary()
    model.compile(loss="categorical_crossentropy", optimizer="Adam", metrics=["accuracy"])
    model.fit(x_train, y_train, epochs=20, validation_split=0.1)
    model.save(MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
