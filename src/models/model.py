from .layer import Layer


class Model:
    def __init__(self, layers: list[Layer] = []):
        self.layers: list[Layer] = layers

    def __repr__(self):
        return f"Model(layers={self.layers})"

    def add_layer(self, layer: Layer):
        self.layers.append(layer)

    def fit(self, x_train: list[float], y_train: list[float]):
        # TODO: Train
        ...
