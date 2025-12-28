from .neuron import Neuron
from .activation import Activation
from .link import Link
from typing import Optional
import random
from math import sqrt


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

    def add_links(self, previous_layer):
        if not len(previous_layer):
            return self
        new_neurons: list[Neuron] = []
        for i in range(len(self)):
            current_neuron = self.neurons[i]
            neuron: Neuron = current_neuron.copyWith(
                input_links=[
                    Link(
                        weight=he_weight(fan_in=len(previous_layer)),
                        source=previous_layer.neurons[prev_idx],
                    )
                    for prev_idx in range(len(previous_layer))
                ]
            )
            new_neurons.append(neuron)

        updated_layer: Layer = self.copyWith(neurons=new_neurons)
        return updated_layer


def he_weight(fan_in):
    sigma: float = sqrt(2 / fan_in)
    mu: float = 0
    random_weight: float = random.gauss(mu=mu, sigma=sigma)
    return random_weight


class Dense(Layer):
    def __init__(
        self,
        number_neurons: int,
        activation: Activation,
    ):
        neurons = [Neuron(bias=0, activation=activation) for _ in range(number_neurons)]
        super().__init__(neurons=neurons)
