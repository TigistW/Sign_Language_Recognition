import cv2
import numpy as np
import tensorflow as tf
import pandas as pd
# Load the model
model = tf.keras.models.load_model('finalmodel')

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

def predict(input_data):
    predictions = model.predict(input_data)
    return predictions


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

        print(predictions)
        cv2.imshow('Camera Feed', frame)

        if cv2.waitKey(1000) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
