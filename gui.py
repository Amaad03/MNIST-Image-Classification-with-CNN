import tkinter as tk
from tkinter import Canvas, Button
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from tensorflow.keras.models import load_model

class DigitRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Digit Recognizer")
        self.root.resizable(0, 0)

        self.canvas = Canvas(self.root, width=200, height=200, bg="white")
        self.canvas.grid(row=0, column=0, pady=2, sticky=tk.W, columnspan=2)
        self.classify_btn = Button(self.root, text="Recognize", command=self.classify_handwriting)
        self.classify_btn.grid(row=1, column=0, pady=2, padx=2)
        self.clear_btn = Button(self.root, text="Clear", command=self.clear_canvas)
        self.clear_btn.grid(row=1, column=1, pady=2, padx=2)

        self.canvas.bind("<B1-Motion>", self.draw_lines)

        self.image = Image.new("L", (200, 200), color="white")
        self.draw = ImageDraw.Draw(self.image)

        # Load the pre-trained model
        self.model = load_model('mnist_model.keras')

    def clear_canvas(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, 200, 200], fill="white")

    def classify_handwriting(self):
        # Resize image to 28x28 pixels
        resized_image = self.image.resize((28, 28))
        inverted_image = ImageOps.invert(resized_image)
        image_array = np.array(inverted_image).astype('float32') / 255.0
        image_array = image_array.reshape(1, 28, 28, 1)

        # Predict the digit
        prediction = self.model.predict(image_array)
        digit = np.argmax(prediction)

        # Print the prediction in the terminal
        print(f"Predicted Digit: {digit}")

    def draw_lines(self, event):
        x, y = event.x, event.y
        self.canvas.create_oval(x, y, x+8, y+8, fill='black', width=10)
        self.draw.ellipse([x, y, x+8, y+8], fill='black')

if __name__ == "__main__":
    root = tk.Tk()
    app = DigitRecognizerApp(root)
    root.mainloop()
