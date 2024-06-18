import tensorflow as tf
# from tensorflow.keras.models import load_model
import numpy as np
import cv2
import pandas as pd

# Load the model in H5 format
model = tf.keras.models.load_model('finalmodel.h5')
# model = tf.keras.models.load_model('themodel')

def preprocess(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Image not found or path is incorrect")
    image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA)
    pixels = image.flatten()
    
    data = {}
    for i, pixel in enumerate(pixels):
        data[f'pixel{i+1}'] = [pixel]

    df = pd.DataFrame(data)
    print(df)
    df=df.values.reshape(-1,28,28,1)
    
    return df

def predict(val):
    res = model.predict(val)

imag = "3001.jpg"
preprocessed = preprocess(imag)
ans = predict(preprocessed)

print(ans)
