from pathlib import Path
from tkinter import Button, Entry, Label, StringVar, Tk, filedialog

import cv2
import numpy as np
from keras.models import load_model


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "brain_tumor_model.h5"
LABELS = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]


def predict(image_path: str) -> str:
    path = Path(image_path)
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model file: {MODEL_PATH}")
    if not path.exists():
        raise FileNotFoundError(f"Missing image file: {path}")

    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Failed to load image: {path}")

    model = load_model(MODEL_PATH)
    img = cv2.resize(img, (150, 150))
    img_array = np.array(img).reshape(1, 150, 150, 3)
    prediction = model.predict(img_array)
    return LABELS[prediction.argmax()]


def select_image() -> None:
    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
    )
    if file_path:
        selected_path.set(file_path)
        result_text.set("")


def run_prediction() -> None:
    try:
        result_text.set(predict(selected_path.get()))
    except Exception as exc:
        result_text.set(str(exc))


window = Tk()
window.title("Brain Tumour Classification")
window.geometry("720x260")
window.configure(bg="#ffffff")

selected_path = StringVar()
result_text = StringVar()

Label(window, text="Image", bg="#ffffff", anchor="w").place(x=24, y=28, width=100, height=28)
Entry(window, textvariable=selected_path).place(x=120, y=28, width=460, height=30)
Button(window, text="Browse", command=select_image).place(x=600, y=28, width=90, height=30)

Button(window, text="Predict", command=run_prediction).place(x=315, y=92, width=90, height=34)

Label(window, text="Prediction", bg="#ffffff", anchor="w").place(x=24, y=162, width=100, height=28)
Label(window, textvariable=result_text, bg="#f2f2f2", anchor="w").place(
    x=120,
    y=162,
    width=570,
    height=34,
)

window.resizable(False, False)
window.mainloop()
