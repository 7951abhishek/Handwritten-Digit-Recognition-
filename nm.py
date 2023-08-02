import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageDraw
import tensorflow as tf
import numpy as np


def recognize_number():
    # Load your trained model
    model = tf.keras.models.load_model('my_mnistnewcolab.h5')

    image = Image.new('L', (280, 280), 'white')
    draw = ImageDraw.Draw(image)
    for point in points:
        draw.ellipse((point[0] - 10, point[1] - 10, point[0] + 10, point[1] + 10), fill='black')

    image = image.resize((28, 28))
    image_data = np.array(image) / 255.0  # Preprocess the image

    # Perform inference
    input_data = np.expand_dims(image_data, axis=0)
    prediction = model.predict(input_data)
    predicted_label = np.argmax(prediction)

    # Display the result
    messagebox.showinfo("Prediction", f"The predicted number is {predicted_label}.")

    canvas.delete('all')
    points.clear()

def draw(event):
    x, y = event.x, event.y
    canvas.create_oval(x - 10, y - 10, x + 10, y + 10, fill='black')
    points.append((x, y))

def create_widgets():
    window = tk.Tk()
    window.title("Handwritten Number Recognition")

    global canvas
    canvas = tk.Canvas(window, width=280, height=280, bg='white')
    canvas.pack()

    global points
    points = []

    canvas.bind("<B1-Motion>", draw)

    recognize_button = tk.Button(window, text='Recognize', command=recognize_number)
    recognize_button.pack()

    window.mainloop()


if __name__ == '__main__':
    create_widgets()
