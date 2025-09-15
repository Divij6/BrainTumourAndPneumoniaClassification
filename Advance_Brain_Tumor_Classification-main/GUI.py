from pathlib import Path
from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage, filedialog
from json import load
from keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import pandas as pd
import numpy as np
import tensorflow as tf
import os




# Load the model
load1 = load_model('brain_tumor_model.h5')

def predict(img1):
    image_path = img1
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file '{image_path}' not found.")

    # Load the image
    img = cv2.imread(image_path)

    # Check if the image loaded successfully
    if img is None:
        raise ValueError(f"Failed to load image '{image_path}'.")

    # Resize the image
    img = cv2.resize(img, (150, 150))

    img_array = np.array(img)
    img_array = img_array.reshape(1, 150, 150, 3)

    a = load1.predict(img_array)
    indices = a.argmax()

    if indices == 0:
        return "glioma_tumour"
    elif indices == 1:
        return "meningioma_tumor"
    elif indices == 2:
        return "no_tumor"
    else:
        return "pituitary_tumor"

def show_image(img):
    from tensorflow.keras.preprocessing import image
    img = image.load_img(img)
    plt.imshow(img, interpolation='nearest')
    plt.show()


OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path(r"E:\Project\Brain Tumour\build\assets\frame0")

def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)

# Global variable to store the file path
selected_file_path = ""

def open_file_explorer():
    global selected_file_path
    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
    )
    if file_path:
        selected_file_path = file_path
        entry_1.delete(0, "end")
        entry_1.insert(0, file_path)

def run_prediction():
    if selected_file_path:
        try:
            # Call the predict function and capture the output
            result = predict(selected_file_path)
            # Display the result in the text box
            entry_2.delete("1.0", "end")
            entry_2.insert("1.0", result)
            show_image(selected_file_path)
        except Exception as e:
            entry_2.delete("1.0", "end")
            entry_2.insert("1.0", str(e))

window = Tk()

window.geometry("1440x1024")
window.configure(bg = "#FFFFFF")

canvas = Canvas(
    window,
    bg = "#FFFFFF",
    height = 1024,
    width = 1440,
    bd = 0,
    highlightthickness = 0,
    relief = "ridge"
)

canvas.place(x = 0, y = 0)
entry_image_1 = PhotoImage(
    file=relative_to_assets("entry_1.png"))
entry_bg_1 = canvas.create_image(
    720.0,
    134.0,
    image=entry_image_1
)
entry_1 = Entry(
    bd=0,
    bg="#A99FE9",
    fg="#000716",
    highlightthickness=0
)
entry_1.place(
    x=104.0,
    y=104.0,
    width=1232.0,
    height=58.0
)

button_image_1 = PhotoImage(
    file=relative_to_assets("button_1.png"))
button_1 = Button(
    image=button_image_1,
    borderwidth=0,
    highlightthickness=0,
    command=open_file_explorer,
    relief="flat"
)
button_1.place(
    x=1275.0,
    y=106.0,
    width=53.0,
    height=56.0
)

button_image_2 = PhotoImage(
    file=relative_to_assets("button_2.png"))
button_2 = Button(
    image=button_image_2,
    borderwidth=0,
    highlightthickness=0,
    command=run_prediction,
    relief="flat"
)
button_2.place(
    x=642.0,
    y=213.0,
    width=191.0,
    height=56.0
)

entry_image_2 = PhotoImage(
    file=relative_to_assets("entry_2.png"))
entry_bg_2 = canvas.create_image(
    745.5,
    429.0,
    image=entry_image_2
)
entry_2 = Text(
    bd=0,
    bg="#D9D9D9",
    fg="#000716",
    highlightthickness=0
)
entry_2.place(
    x=431.0,
    y=391.0,
    width=629.0,
    height=74.0
)

window.resizable(True, True)
window.mainloop()
