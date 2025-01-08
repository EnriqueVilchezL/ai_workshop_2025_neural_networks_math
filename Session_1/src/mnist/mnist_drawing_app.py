import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from typing import List
from network.sequential import Sequential

class MNISTDrawingApp:
    def __init__(self, model : Sequential):
        
        self.model : Sequential = model
        self.GRID_SIZE : int = 28
        self.CELL_SIZE : int = 20
        self.grid : np.ndarray = np.zeros((self.GRID_SIZE, self.GRID_SIZE))

        # Create the main Tkinter window
        self.root : tk.Tk = tk.Tk()
        self.root.title("MNIST Drawing App")

        # Create the canvas for drawing
        self.canvas : tk.Canvas = tk.Canvas(
            self.root,
            width=self.GRID_SIZE * self.CELL_SIZE,
            height=self.GRID_SIZE * self.CELL_SIZE,
            bg="black"
        )
        self.canvas.pack()

        # Bind mouse events for drawing
        self.canvas.bind("<B1-Motion>", self.draw)

        # Add buttons
        self.button_frame : tk.Frame = tk.Frame(self.root)
        self.button_frame.pack()

        self.predict_button : tk.Button = tk.Button(
            self.button_frame, text="Predict", command=self.predict_digit
        )
        self.predict_button.pack(side="left", padx=10)

        self.clear_button : tk.Button = tk.Button(
            self.button_frame, text="Clear", command=self.clear_grid
        )
        self.clear_button.pack(side="left", padx=10)

        # Label to display prediction
        self.result_label : tk.Label = tk.Label(self.root, text="Draw a digit and press Predict", font=("Helvetica", 16))
        self.result_label.pack(pady=10)

    def draw(self, event : tk.Event) -> None:
        x, y = event.x, event.y
        row, col = y // self.CELL_SIZE, x // self.CELL_SIZE

        if 0 <= row < self.GRID_SIZE and 0 <= col < self.GRID_SIZE:
            # Draw on the canvas
            self.canvas.create_rectangle(
                col * self.CELL_SIZE, row * self.CELL_SIZE,
                (col + 1) * self.CELL_SIZE, (row + 1) * self.CELL_SIZE,
                fill="white"
            )
            # Update the grid
            self.grid[row, col] = 255

    def clear_grid(self):
        self.canvas.delete("all")
        self.grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE))
        self.result_label.config(text="Draw a digit and press Predict")

    def predict_digit(self):
        input_data = self.grid.flatten().reshape(1, -1)
        input_data = input_data.astype("float32") / 255.0
        prediction = self.model({'X': input_data})
        self.result_label.config(text=f"Predicted Digit: {np.argmax(prediction).sum()}")

    def run(self):
        self.root.mainloop()
