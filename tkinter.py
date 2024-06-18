import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageTk

# Load the model
model = tf.keras.models.load_model('finalmodel')

# Initialize variables
is_recording = False
cap = None

def preprocess(image):
    resized = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    normalized = gray / 255.0
    reshaped = normalized.reshape(-1, 28, 28, 1)
    return reshaped

def predict(input_data):
    predictions = model.predict(input_data)
    return predictions

def start_recording():
    global is_recording, cap
    if not is_recording:
        cap = cv2.VideoCapture(0)  # Start camera capture
        is_recording = True
        capture_video()

def stop_recording():
    global is_recording, cap
    if is_recording:
        cap.release()  # Release camera
        is_recording = False

def capture_video():
    global cap
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            preprocessed = preprocess(frame)
            predictions = predict(preprocessed)
            print(predictions)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            video_canvas.imgtk = imgtk
            video_canvas.configure(image=imgtk)

    if is_recording:
        video_canvas.after(10, capture_video)

root = tk.Tk()
root.title("Camera Interface")

video_canvas = tk.Label(root)
video_canvas.pack(padx=10, pady=10)

control_frame = ttk.Frame(root)
control_frame.pack(padx=10, pady=10)

start_button = ttk.Button(control_frame, text="Start Recording", command=start_recording)
start_button.grid(row=0, column=0, padx=5, pady=5)

stop_button = ttk.Button(control_frame, text="Stop Recording", command=stop_recording)
stop_button.grid(row=0, column=1, padx=5, pady=5)

root.mainloop()

if cap is not None:
    cap.release()
cv2.destroyAllWindows()
