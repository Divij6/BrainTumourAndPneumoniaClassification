from json import load
from keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import pandas as pd
import numpy as np
import tensorflow as tf 
from tensorflow.keras.preprocessing import image
import os 
load = load_model('brain_tumor_model.h5')



image_path = "Advance_Brain_Tumor_Classification-main/download.jpeg"
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
img_array.shape
img_array = img_array.reshape(1,150,150,3)
img_array.shape

img = image.load_img("Advance_Brain_Tumor_Classification-main/download.jpeg")


a=load.predict(img_array)
indices = a.argmax()
print (indices)

if(indices==0):
    print("Glioma Tumor")
if(indices==1):
    print("Meningioma Tumor")
if(indices==2):
    print("No Tumor")
if(indices==3):
    print("Pituitary Tumor")
plt.imshow(img,interpolation='nearest')
plt.show()