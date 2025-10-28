import tkinter as tk
from tkinter import Button, Label
import tensorflow as tf
from PIL import Image, ImageDraw
import numpy as np
from scipy import ndimage
import io

try:
    model = tf.keras.models.load_model('digit_recognizer_model.keras')
except IOError:
    print("Model not found! Please run main.py first to train and save the model.")
    exit()

def clear_canvas():
    canvas.delete("all")
    global pil_image, pil_draw
    pil_image.paste(0, [0, 0, pil_image.size[0], pil_image.size[1]])
    prediction_label.config(text="Prediction: -")

def process_image(img):
    data = np.array(img)
    if not np.any(data): return None
    rows = np.any(data, axis=1)
    cols = np.any(data, axis=0)
    if not np.any(rows) or not np.any(cols): return None
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    cropped = data[ymin:ymax+1, xmin:xmax+1]
    rows, cols = cropped.shape
    if rows > cols: factor = 20.0 / rows; rows = 20; cols = int(round(cols * factor))
    else: factor = 20.0 / cols; cols = 20; rows = int(round(rows * factor))
    cropped_pil = Image.fromarray(cropped)
    resized = cropped_pil.resize((cols, rows), Image.Resampling.LANCZOS)
    resized_data = np.array(resized)
    box = np.zeros((28, 28), dtype=np.uint8)
    x_offset = (28 - cols) // 2; y_offset = (28 - rows) // 2
    box[y_offset:y_offset + rows, x_offset:x_offset + cols] = resized_data
    cy, cx = ndimage.center_of_mass(box)
    rows, cols = box.shape
    shiftx = np.round(cols/2.0 - cx).astype(int)
    shifty = np.round(rows/2.0 - cy).astype(int)
    centered_box = ndimage.shift(box, [shifty, shiftx])
    return centered_box

def predict_digit():
    img = pil_image.copy()
    processed_img_array = process_image(img)
    if processed_img_array is None:
        prediction_label.config(text="Prediction: - (Canvas empty)")
        return
    final_image = processed_img_array / 255.0
    final_image = final_image.reshape(1, 28, 28, 1)
    prediction = model.predict(final_image)
    predicted_digit = np.argmax(prediction)
    prediction_label.config(text=f"Prediction: {predicted_digit}")

def start_drawing(event):
    global last_x, last_y
    last_x, last_y = event.x, event.y

def draw(event):
    global last_x, last_y
    canvas.create_line((last_x, last_y, event.x, event.y), fill='white', width=25, capstyle=tk.ROUND, smooth=tk.TRUE)
    pil_draw.line((last_x, last_y, event.x, event.y), fill=255, width=25)
    last_x, last_y = event.x, event.y

window = tk.Tk()
window.title("Digit Recognizer")
canvas = tk.Canvas(window, width=280, height=280, bg="black", cursor="cross")
prediction_label = Label(window, text="Prediction: -", font=("Helvetica", 20))
predict_button = Button(window, text="Predict", command=predict_digit, font=("Helvetica", 14))
clear_button = Button(window, text="Clear", command=clear_canvas, font=("Helvetica", 14))
canvas.grid(row=0, column=0, columnspan=2, pady=10, padx=10)
prediction_label.grid(row=1, column=0, columnspan=2)
predict_button.grid(row=2, column=0, pady=10, padx=10, sticky="ew")
clear_button.grid(row=2, column=1, pady=10, padx=10, sticky="ew")
canvas.bind("<Button-1>", start_drawing)
canvas.bind("<B1-Motion>", draw)
pil_image = Image.new('L', (280, 280), color=0)
pil_draw = ImageDraw.Draw(pil_image)

window.mainloop()
