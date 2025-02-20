import tkinter as tk
from network.sequential import Sequential
from network.layer import Dense
from network.activation import Activation
import numpy as np

class NeuralNetworkVisualizer:
    """Handles the visualization of neural network activations."""
    
    def __init__(self, root, model: Sequential, width=400, height=500):
        self.root = root
        self.model = model
        self.layers = self.get_model_dense_layers()
        self.node_positions = {}
        self.neurons = {}

        # Set initial canvas size
        self.canvas_width = width
        self.canvas_height = height
        self.canvas = tk.Canvas(root, width=self.canvas_width, height=self.canvas_height, bg="white")
        self.canvas.pack(side="right", padx=10, pady=10, fill="both", expand=True)

        # Bind the resize event of the canvas to handle scaling
        self.canvas.bind("<Configure>", self.on_canvas_resize)

        # Initial drawing of the network
        self.draw_network()

    def get_model_dense_layers(self):
        layers = []
        stack = [self.model]
        module_count = 0

        while stack:
            current_module = stack.pop()
            for submodule in current_module.submodules:
                if submodule.submodules is not None:
                    stack.append(submodule)
                
                if isinstance(submodule, Dense):
                    module_count += 1
                    if module_count == 1:
                        layers.append(submodule.input_size)

                    layers.append(submodule.output_size)
        
        return layers

    def get_model_activations(self):
        activations = []
        stack = [self.model]
        module_count = 0

        while stack:
            current_module = stack.pop()
            for submodule in current_module.submodules:
                if submodule.submodules is not None:
                    stack.append(submodule)
                
                if isinstance(submodule, Dense):
                    module_count += 1
                    if module_count == 1:
                        activations.append(submodule.tensors['X'])
                    
                if isinstance(submodule, Activation):
                    activations.append(submodule.tensors['Y'])
        
        return activations
    
    def draw_network(self):
        # Use the current canvas size
        width, height = self.canvas.winfo_width(), self.canvas.winfo_height()

        num_layers = len(self.layers)
        layer_spacing = width // (num_layers + 1)
        for i, num_nodes in enumerate(self.layers):
            x = (i + 1) * layer_spacing
            y_spacing = max(height // (num_nodes + 1), 23)
            self.node_positions[i] = []
            self.neurons[i] = []

            if num_nodes > 20:
                for j in range(10):
                    y = (j + 1) * y_spacing
                    node = self.canvas.create_oval(
                        x - 8, y - 8, x + 8, y + 8, fill="gray", outline="black"
                    )
                    self.neurons[i].append(node)
                    self.node_positions[i].append((x, y))
                
                # Draw 3 vertical suspensive dots for hidden nodes
                self.canvas.create_text(x, y_spacing * 11, text=".", fill="gray")
                self.canvas.create_text(x, y_spacing * 12, text=".", fill="gray")
                self.canvas.create_text(x, y_spacing * 13, text=".", fill="gray")

                for j in range(13, 23):
                    y = (j + 1) * y_spacing
                    node = self.canvas.create_oval(
                        x - 8, y - 8, x + 8, y + 8, fill="gray", outline="black"
                    )
                    self.neurons[i].append(node)
                    self.node_positions[i].append((x, y))
                
            else:
                for j in range(num_nodes):
                    y = (j + 1) * y_spacing
                    node = self.canvas.create_oval(
                        x - 8, y - 8, x + 8, y + 8, fill="gray", outline="black"
                    )
                    self.neurons[i].append(node)
                    self.node_positions[i].append((x, y))

        # Draw connections
        for i in range(num_layers - 1):
            for pos1 in self.node_positions[i]:
                for pos2 in self.node_positions[i + 1]:
                    self.canvas.create_line(
                        pos1[0] + 8, pos1[1], pos2[0] - 8, pos2[1], fill="gray", width=1
                    )

    def on_canvas_resize(self, event):
        # Clear the canvas to avoid overlapping drawings
        self.canvas.delete("all")

        # Recalculate and redraw the network on the new canvas size
        self.draw_network()

    def reset_network(self):
        for layer in self.neurons:
            for neuron in self.neurons[layer]:
                self.canvas.itemconfig(neuron, fill="gray")

    def animate_activations(self):
        self.reset_network()
        self.activations = self.get_model_activations()
        self.current_layer = 0
        self.current_neuron = 0
        self.animate_next_layer()

    def animate_next_layer(self):
        if self.current_layer < len(self.layers):
            if self.current_neuron < min(self.layers[self.current_layer], 20):
                activation_value = self.activations[self.current_layer][0, self.current_neuron]
                self.animate_neuron_activation(self.current_layer, self.current_neuron, activation_value, 20, 200)
                self.current_neuron += 1
                self.root.after(20, self.animate_next_layer)
            else:
                self.current_layer += 1
                self.current_neuron = 0
                self.root.after(200, self.animate_next_layer)

    def animate_neuron_activation(self, layer, neuron_index, activation, steps=20, duration=200):
        start_color = (169, 169, 169)
        target_color = (255, 255, 0)

        delta_r = (target_color[0] - start_color[0]) * activation / steps
        delta_g = (target_color[1] - start_color[1]) * activation / steps
        delta_b = (target_color[2] - start_color[2]) * activation / steps

        def clamp(value):
            return max(0, min(255, int(value)))

        def step(count):
            if count <= steps:
                new_r = clamp(start_color[0] + delta_r * count)
                new_g = clamp(start_color[1] + delta_g * count)
                new_b = clamp(start_color[2] + delta_b * count)
                new_color = f'#{new_r:02x}{new_g:02x}{new_b:02x}'
                self.canvas.itemconfig(self.neurons[layer][neuron_index], fill=new_color)
                self.root.after(int(duration / steps), lambda: step(count + 1))

        step(0)
