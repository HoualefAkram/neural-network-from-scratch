from .neuron import Neuron
from .activation import Activation
from typing import Optional


class Layer:
    """
    Parent class of layers, a layer must have at least 1 neuron
    """

    def __init__(self, neurons: list[Neuron]):
        if not neurons:
            raise ValueError("A Layer must have at least one neuron")
        self.neurons: list[Neuron] = neurons

    def __repr__(self):
        return f"Layer(neurons={self.neurons})"

    def __len__(self):
        return len(self.neurons)

    def copyWith(self, neurons: Optional[list[Neuron]] = None):
        return Layer(neurons=self.neurons if (neurons is None) else neurons)


class Dense(Layer):
    def __init__(
        self,
        number_neurons: int,
        activation: Activation,
    ):
        neurons = [Neuron(bias=0, activation=activation) for _ in range(number_neurons)]
        super().__init__(neurons=neurons)
