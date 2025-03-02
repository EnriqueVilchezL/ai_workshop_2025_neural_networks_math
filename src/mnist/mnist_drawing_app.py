import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from network.sequential import Sequential
from mnist.neural_network_drawing_app import NeuralNetworkVisualizer

class MNISTDrawingApp:
    def __init__(self, model : Sequential):
        self.model: Sequential = model
        self.GRID_SIZE: int = 28
        self.CELL_SIZE: int = 20
        self.grid: np.ndarray = np.zeros((self.GRID_SIZE, self.GRID_SIZE))

        # Create the main Tkinter window
        self.root: tk.Tk = tk.Tk()
        self.root.title("MNIST Drawing App")

        # Create a frame for both the drawing area and the neural network visualizer
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill="both", expand=True)

        self.content_frame = tk.Frame(self.main_frame)
        self.content_frame.pack(side="bottom", padx=10, pady=10)

        # Create a frame for the left side canvas (drawing area)
        self.canvas_frame = tk.Frame(self.content_frame)
        self.canvas_frame.pack(side="left", padx=10, pady=10)

        # Create the canvas for drawing
        self.canvas: tk.Canvas = tk.Canvas(
            self.canvas_frame,
            width=self.GRID_SIZE * self.CELL_SIZE,
            height=self.GRID_SIZE * self.CELL_SIZE,
            bg="black"
        )
        self.canvas.pack()

        # Bind mouse events for drawing
        self.canvas.bind("<B1-Motion>", self.draw)

        self.decorations_frame = tk.Frame(self.main_frame)
        self.decorations_frame.pack(side="top", pady=10)

        # Add the label above the buttons
        self.result_label: tk.Label = tk.Label(self.decorations_frame, text="Draw a digit and press Predict", font=("Helvetica", 16))
        self.result_label.pack(side="top", pady=10)

        # Add buttons below the label
        self.button_frame: tk.Frame = tk.Frame(self.decorations_frame)
        self.button_frame.pack(side="top", pady=10)

        self.predict_button: tk.Button = tk.Button(
            self.button_frame, text="Predict", command=self.predict_digit
        )
        self.predict_button.pack(side="left", padx=10)

        self.clear_button: tk.Button = tk.Button(
            self.button_frame, text="Clear", command=self.clear_grid
        )
        self.clear_button.pack(side="left", padx=10)

        # Create a frame for the right side (visualizer)
        self.visualizer_frame = tk.Frame(self.content_frame)
        self.visualizer_frame.pack(side="right", padx=10, pady=10)

        # Initialize NeuralNetworkVisualizer on the right side
        self.visualizer: NeuralNetworkVisualizer = NeuralNetworkVisualizer(
            self.visualizer_frame,
            self.model,
            self.GRID_SIZE * self.CELL_SIZE,
            self.GRID_SIZE * self.CELL_SIZE
        )

    def draw(self, event: tk.Event) -> None:
        x, y = event.x, event.y
        row, col = y // self.CELL_SIZE, x // self.CELL_SIZE

        # Define relative neighbor positions (including the center cell)
        neighbors = [(0,0)]

        for dr, dc in neighbors:
            nr, nc = row + dr, col + dc  # Compute neighbor coordinates

            if 0 <= nr < self.GRID_SIZE and 0 <= nc < self.GRID_SIZE:
                self.grid[nr, nc] = 255
                
                gray_value = int(self.grid[nr, nc])  # Get grayscale intensity (0-255)
                color = f'#{gray_value:02x}{gray_value:02x}{gray_value:02x}'

                # Draw on the canvas
                self.canvas.create_rectangle(
                    nc * self.CELL_SIZE, nr * self.CELL_SIZE,
                    (nc + 1) * self.CELL_SIZE, (nr + 1) * self.CELL_SIZE,
                    fill=color, outline=""
                )

    def clear_grid(self):
        self.canvas.delete("all")
        self.grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE))
        self.result_label.config(text="Draw a digit and press Predict")
        self.visualizer.reset_network()

    def predict_digit(self):
        input_data = self.grid.flatten().reshape(1, -1)
        input_data = input_data.astype("float32") / 255.0
        prediction = self.model({'X': input_data})
        self.result_label.config(text=f"Predicted Digit: {np.argmax(prediction).sum()}")
        self.visualizer.animate_activations()


    def run(self):
        self.root.mainloop()
