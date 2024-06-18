import cv2
import numpy as np
import pickle
import tensorflow as tf
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense,Flatten,Conv2D,MaxPool2D,Dropout
# Load the model in H5 format

model=Sequential()
# model.add()
model.add(Conv2D(128,kernel_size=(5,5),
                 strides=1,padding='same',activation='relu',input_shape=(28,28,1)))
model.add(MaxPool2D(pool_size=(3,3),strides=2,padding='same'))
model.add(Conv2D(64,kernel_size=(2,2),
                strides=1,activation='relu',padding='same'))
model.add(MaxPool2D((2,2),2,padding='same'))
model.add(Conv2D(32,kernel_size=(2,2),
                strides=1,activation='relu',padding='same'))
model.add(MaxPool2D((2,2),2,padding='same'))
          
model.add(Flatten())
model.add(Dense(units=512,activation='relu'))
model.add(Dropout(rate=0.25))
model.add(Dense(units=28,activation='softmax'))
# model.summary()
model.load_weights("modelh5.h5")
# model = tf.keras.models.load_model('finalmodel.h5')

def preprocess(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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
    return res

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break
        preprocessed = preprocess(frame)
        predictions = predict(preprocessed)

        # print(predictions)

        # Load the LabelBinarizer from the saved file
        with open('label_binarizer.pkl', 'rb') as f:
            lb = pickle.load(f)
        original_labels = lb.inverse_transform(predictions)
        print(original_labels)

        cv2.imshow('Camera Feed', frame)

        if cv2.waitKey(1000) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
