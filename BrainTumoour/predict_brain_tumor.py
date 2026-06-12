from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
from tensorflow.keras.preprocessing import image


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "brain_tumor_model.h5"
DEFAULT_IMAGE_PATH = BASE_DIR / "sample_images" / "download.jpeg"
LABELS = ["Glioma Tumor", "Meningioma Tumor", "No Tumor", "Pituitary Tumor"]


def predict(image_path: Path) -> str:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model file: {MODEL_PATH}")
    if not image_path.exists():
        raise FileNotFoundError(f"Missing image file: {image_path}")

    model = load_model(MODEL_PATH)
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")

    img = cv2.resize(img, (150, 150))
    img_array = np.array(img).reshape(1, 150, 150, 3)
    prediction = model.predict(img_array)
    return LABELS[prediction.argmax()]


def main() -> None:
    result = predict(DEFAULT_IMAGE_PATH)
    print(result)

    img = image.load_img(DEFAULT_IMAGE_PATH)
    plt.imshow(img, interpolation="nearest")
    plt.title(result)
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()
