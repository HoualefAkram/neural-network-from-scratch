from typing import Callable
from .activation import Activation


class Neuron:

    def __init__(self, bias: float, activation: Activation):
        self.bias: float = bias
        self.activation: Activation = activation

    def __repr__(self):
        return f"Neuron(bias={self.bias},activation={type(self.activation)})"
