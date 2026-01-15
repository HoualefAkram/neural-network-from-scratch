# Neural Network from Scratch (Pure Object-Oriented Python)

This repository contains a sophisticated, "from-scratch" implementation of a Deep Neural Network. This project is unique because it avoids matrix-math libraries like NumPy in favor of a Pure Object-Oriented approach using Neurons, Links, and Layers as distinct Python objects.

---

## 🚀 Overview

Most neural networks are implemented using optimized matrix multiplication. This project takes a different path by building a Computational Graph where every connection (Link) and every node (Neuron) is an individual object. This provides a granular view of how data and gradients flow through a network.

### Key Features
* **Zero 3rd-Party Math Libraries**: Built using only Python's `math` and `random` modules.
* **True Object-Oriented Design**:
    - **Neuron**: Manages its own bias, activation, and input summation.
    - **Link**: Manages the weight between two specific neurons.
    - **Layer**: Organizes neurons into Input, Dense, and Output structures.
* **Advanced Initializers**: Implements **He Initialization** (Kaiming Init) using Gaussian distribution to prevent vanishing/exploding gradients.
* **Custom Activations**: Includes manual implementations of **ReLU**, **SoftPlus**, and **Linear** activations.
* **Flexible Architecture**: Support for an arbitrary number of hidden layers and neurons.

---

## 📂 Project Structure

* **src/models/neuron.py**: The core atomic unit. Handles the weighted sum calculation.
* **src/models/link.py**: Represents the synaptic connection between neurons.
* **src/models/layer.py**: Handles layer stacking and automatic link creation (Dense/Input).
* **src/models/activation.py**: Implementation of activation functions (SoftPlus, ReLU, Linear).
* **src/models/error.py**: Error calculation logic (Mean Squared Error).
* **main.py**: Entry point to build, train, and visualize the model.

---

## 🛠️ Installation & Requirements

1. **Clone the repository**:
   git clone https://github.com/HoualefAkram/neural-network-from-scratch.git
   cd neural-network-from-scratch

2. **Requirements**:
   - Python 3.x
   - Matplotlib (optional, only for plotting results in main.py)

---

## 💻 Technical Highlights

### 1. The Neuron Graph
Instead of `Y = WX + B`, this model calculates values by traversing the graph:
z = sum(link.weight * link.source.get_value() for link in self.input_links) + self.bias

### 2. He Initialization
To ensure the model trains effectively from the start, weights are initialized based on the "fan-in" (number of inputs) of the layer:
sigma = sqrt(2 / fan_in)
weight = random.gauss(mu=0, sigma=sigma)

### 3. SoftPlus Activation
The model uses the SoftPlus function: f(x) = ln(1 + exp(x)), which is a smooth approximation of the ReLU function.

---

## 📊 Example Usage

# Define the architecture
model = Model()
model.add_layer(layer=Input(number_features=1))
model.add_layer(layer=Dense(4, activation=SoftPlus()))
model.add_layer(layer=Dense(4, activation=SoftPlus()))
model.add_layer(layer=Dense(1, activation=Linear()))

# Train using pure Python loops
model.fit(x_train, y_train, iterations=15000, learning_rate=0.1)

---

## 🧠 Why This Matters

This "Pure OO" approach is rarely seen because it is computationally more expensive than NumPy matrices. However, it is the best way to:
- Understand exactly how **backpropagation** affects individual weights.
- Visualize the network as a literal graph of connected objects.
- Debug the math of a single neuron without looking at a giant multi-dimensional array.

---

## 📜 License

Distributed under the MIT License.

**Author:** [Houalef Akram](https://github.com/HoualefAkram)