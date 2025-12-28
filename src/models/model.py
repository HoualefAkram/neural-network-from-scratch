from .layer import Layer


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

    def fit(self, x_train: list[float], y_train: list[float]):
        # TODO: Train
        ...

    def predict(self, x_test: list[float]):
        # 1- feed the input to the first layer
        first_layer: Layer = self.layers[0]
        if len(x_test) != len(first_layer):
            raise ValueError("length of x_test must be equal to the first layer length")
        for i in range(len(x_test)):
            first_layer.neurons[i].set_value(x_test[i])
        # 2- forward passing
