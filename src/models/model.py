from typing import Optional
from .layer import Layer
from .error import Mse
from .neuron import Neuron


class Model:

    def __init__(self, layers: Optional[list[Layer]] = None):
        self.layers: list[Layer] = layers
        self.__h = 1e-6

    def copyWith(self, layers: Optional[list[Layer]] = None):
        return Model(layers=list(self.layers) if layers is None else layers)

    def __repr__(self):
        return f"Model(layers={self.layers})"

    def add_layer(self, layer: Layer):
        if not self.layers:
            self.layers = []
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

    def __update_tree_from_neuron(
        self,
        neuron: Neuron,
        new_weights: list[float],
        new_biases: list[float],
        updated_neurons_ids: list[id],
    ):
        if neuron.input_links and neuron.id not in updated_neurons_ids:
            updated_neurons_ids.append(neuron.id)
            neuron.bias = new_biases.pop(0)
            for il in neuron.input_links:
                il.weight = new_weights.pop(0)
                self.__update_tree_from_neuron(
                    neuron=il.source,
                    new_weights=new_weights,
                    new_biases=new_biases,
                    updated_neurons_ids=updated_neurons_ids,
                )

    def __update_model_params(self, new_weights: list[float], new_biases: list[float]):
        last_layer: Layer = self.layers[-1]
        for neuron in last_layer.neurons:
            self.__update_tree_from_neuron(
                neuron=neuron,
                new_weights=new_weights,
                new_biases=new_biases,
                updated_neurons_ids=[],
            )

    def __optimize_neuron_tree_once(
        self,
        neuron: Neuron,
        x_train: list[list[float]],
        y_train: list[list[float]],
        learning_rate: float,
        traversed_neurons_ids: list[int],
        output_biases: list[float],
        output_weights: list[list],
    ):
        if neuron.input_links and neuron.id not in traversed_neurons_ids:
            traversed_neurons_ids.append(neuron.id)
            # optimize bias with once
            mse_before = self.__get_mse(inputs=x_train, outputs=y_train)
            neuron.bias += self.__h
            mse_after = self.__get_mse(inputs=x_train, outputs=y_train)
            neuron.bias -= self.__h
            gradient = (mse_after - mse_before) / self.__h
            new_bias = neuron.bias - gradient * learning_rate
            output_biases.append(new_bias)
            # optimize input weights once
            for il in neuron.input_links:
                mse_before = self.__get_mse(inputs=x_train, outputs=y_train)
                il.weight += self.__h
                mse_after = self.__get_mse(inputs=x_train, outputs=y_train)
                il.weight -= self.__h
                gradient = (mse_after - mse_before) / self.__h
                new_weight = il.weight - gradient * learning_rate
                output_weights.append(new_weight)
                # move to the next neuron attached to the link
                self.__optimize_neuron_tree_once(
                    neuron=il.source,
                    x_train=x_train,
                    y_train=y_train,
                    learning_rate=learning_rate,
                    traversed_neurons_ids=traversed_neurons_ids,
                    output_biases=output_biases,
                    output_weights=output_weights,
                )
        else:
            return output_weights, output_biases

    def fit(
        self,
        x_train: list[list[float]],
        y_train: list[list[float]],
        iterations: int,
        learning_rate: float,
    ):
        # TODO: Train
        mse_before = self.__get_mse(inputs=x_train, outputs=y_train)
        last_layer: Layer = self.layers[-1]
        for i in range(iterations):
            print(f"iteration: {i}/{iterations}")
            output_biases = []
            output_weights = []
            for neuron in last_layer.neurons:
                self.__optimize_neuron_tree_once(
                    neuron=neuron,
                    x_train=x_train,
                    y_train=y_train,
                    learning_rate=learning_rate,
                    traversed_neurons_ids=[],
                    output_biases=output_biases,
                    output_weights=output_weights,
                )
            self.__update_model_params(
                new_weights=output_weights, new_biases=output_biases
            )
        mse_after = self.__get_mse(inputs=x_train, outputs=y_train)
        print(f"mse_before: {mse_before}, mse_after: {mse_after}")

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
