from .layer import Layer
from .neuron import Neuron
from .error import Mse


class Model:
    def __init__(self, layers: list[Layer] = []):
        self.layers: list[Layer] = layers

    def __repr__(self):
        return f"Model(layers={self.layers})"

    def add_layer(self, layer: Layer):
        if not self.layers:
            self.layers.append(layer)
            return
        last_layer: Layer = self.layers[-1]
        updated_layer: Layer = layer.add_links(previous_layer=last_layer)
        self.layers.append(updated_layer)

    @property
    def shape(self):
        # return (weight_count, bias_count)
        b_count = sum(
            len(layer) for layer in self.layers if layer.neurons[0].bias is not None
        )
        w_count = sum(
            len(self.layers[i]) * len(self.layers[i + 1])
            for i in range(len(self.layers) - 1)
        )
        return w_count, b_count

    def __traverse_neuron_backward(self, neuron: Neuron):
        if neuron.input_links:
            for il in neuron.input_links:
                self.__traverse_neuron_backward(il.source)

    def fit(self, x_train: list[float], y_train: list[float]):
        # TODO: Train (Backpropagation)
        # go to last Neuron, find its input_links, for each input link to the same until reaching the start point
        last_layer_neurons = self.layers[-1].neurons
        for neuron in last_layer_neurons:
            self.__traverse_neuron_backward(neuron=neuron)

    def predict(self, x_test: list[float]) -> list[float]:
        # 1- feed the input to the first layer
        first_layer: Layer = self.layers[0]
        if len(x_test) != len(first_layer):
            raise ValueError("length of x_test must be equal to the first layer length")
        for i in range(len(x_test)):
            first_layer.neurons[i].set_value(x_test[i])
        # 2- forward passing
        last_layer_neurons = self.layers[-1].neurons
        return [lln.get_value() for lln in last_layer_neurons]
