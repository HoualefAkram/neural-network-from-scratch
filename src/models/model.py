from .layer import Layer
from .link import Link
from .neuron import Neuron
import random
from math import sqrt


class Model:
    def __init__(self, layers: list[Layer] = []):
        self.layers: list[Layer] = layers

    def __repr__(self):
        return f"Model(layers={self.layers})"

    def add_layer(self, layer: Layer):
        # check previous layer to add links
        #  for the first layer, no need to check for previous layers
        if not self.layers:
            self.layers.append(layer)
            return
        last_layer: Layer = self.layers[-1]
        new_neurons: list[Neuron] = []
        for i in range(len(layer)):
            neuron: Neuron = layer.neurons[i].copyWith(
                input_links=[
                    Link(weight=he_weight(layer_len=len(layer)))
                    for _ in range(len(last_layer))
                ]
            )
            new_neurons.append(neuron)

        updated_layer: Layer = layer.copyWith(neurons=new_neurons)
        self.layers.append(updated_layer)

    def fit(self, x_train: list[float], y_train: list[float]):
        # TODO: Train
        ...

    def predict(self, x_test: float):  # TODO: Prediction
        for l in self.layers:
            for n in l.neurons:
                ...


def he_weight(layer_len):
    sigma: float = sqrt(2 / layer_len)
    mu: float = 0
    random_weight: float = random.gauss(mu=mu, sigma=sigma)
    return random_weight
