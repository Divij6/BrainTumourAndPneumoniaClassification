import numpy as np
import pandas as pd 
import os
import keras
from keras.models import Sequential
from keras.layers import Conv2D,Flatten,Dense,MaxPooling2D,Dropout
from sklearn.metrics import accuracy_score
import ipywidgets as widgets
import io
from PIL import Image
import tqdm
from sklearn.model_selection import train_test_split
import cv2
from sklearn.utils import shuffle
import tensorflow as tf

X_train = []
Y_train = []
image_size = 150
labels = ['glioma_tumor','meningioma_tumor','no_tumor','pituitary_tumor']
for i in labels:
    folderPath = os.path.join('Advance_Brain_Tumor_Classification-main/Training', i)
    for j in os.listdir(folderPath):
        try:
            img = cv2.imread(os.path.join(folderPath, j))
            if img is not None:
                img = cv2.resize(img, (image_size, image_size))
                X_train.append(img)
                Y_train.append(i)
            else:
                print(f"Error: Failed to load image: {os.path.join(folderPath, j)}")
        except Exception as e:
            print(f"Error: {e}")
        
for i in labels:
    folderPath = os.path.join('Advance_Brain_Tumor_Classification-main/Testing', i)
    for j in os.listdir(folderPath):
        try:
            img = cv2.imread(os.path.join(folderPath, j))
            if img is not None:
                img = cv2.resize(img, (image_size, image_size))
                X_train.append(img)
                Y_train.append(i)
            else:
                print(f"Error: Failed to load image: {os.path.join(folderPath, j)}")
        except Exception as e:
            print(f"Error: {e}")

X_train = np.array(X_train)
Y_train = np.array(Y_train)


X_train,Y_train = shuffle(X_train,Y_train,random_state=101)
X_train.shape

X_train,X_test,y_train,y_test = train_test_split(X_train,Y_train,test_size=0.1,random_state=101)

y_train_new = []
for i in y_train:
    y_train_new.append(labels.index(i))
y_train=y_train_new
y_train = tf.keras.utils.to_categorical(y_train)

y_test_new = []
for i in y_test:
    y_test_new.append(labels.index(i))
y_test=y_test_new
y_test = tf.keras.utils.to_categorical(y_test)

model = Sequential()
model.add(Conv2D(32,(3,3),activation = 'relu',input_shape=(150,150,3)))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.3))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(Dropout(0.3))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.3))
model.add(Conv2D(128,(3,3),activation='relu'))
model.add(Conv2D(128,(3,3),activation='relu'))
model.add(Conv2D(128,(3,3),activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.3))
model.add(Conv2D(128,(3,3),activation='relu'))
model.add(Conv2D(256,(3,3),activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(512,activation = 'relu'))
model.add(Dense(512,activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(4,activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])

history = model.fit(X_train,y_train,epochs=20,validation_split=0.1)
# Save the model to a file
model.save('brain_tumor_model.h5')
