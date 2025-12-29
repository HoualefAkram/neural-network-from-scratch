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

    def __predict_multiple(self, inputs: list[list[float]]) -> list[list[float]]:
        return [self.predict(x) for x in inputs]

    def __get_mse(self, inputs: list[list[float]], outputs: list[list[float]]):
        predicted_outputs = self.__predict_multiple(inputs)
        return Mse.calc(predicted_outputs=predicted_outputs, outputs=outputs)

    def fit(self, x_train: list[list[float]], y_train: list[list[float]]):
        # TODO: Train (Backpropagation)
        h = 1e-6
        lr = 1e-2
        b3_neuron = self.layers[-1].neurons[0]
        for i in range(1000):
            mse_before = self.__get_mse(inputs=x_train, outputs=y_train)
            b3_neuron.bias += h
            mse_after = self.__get_mse(inputs=x_train, outputs=y_train)
            b3_neuron.bias -= h
            gradient = (mse_after - mse_before) / h
            b3_neuron.bias = b3_neuron.bias - gradient * lr
            if i == 0 or i == 999:
                print(f"mse({i}): {mse_before}")

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
