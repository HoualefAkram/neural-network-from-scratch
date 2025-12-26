from .activation import Activation
from .link import Link
from typing import Optional


class Neuron:

    def __init__(
        self,
        bias: float,
        activation: Activation,
        input_links: Optional[list[Link]] = [],
    ):
        self.bias: float = bias
        self.activation: Activation = activation
        self.input_links: list[Link] = input_links

    @property
    def value(self):
        input_weights: list[float] = [il.weight for il in self.input_links]
        return self.activation.call((sum(input_weights) + self.bias))

    def __repr__(self):
        return f"Neuron(bias={self.bias},activation={type(self.activation)}, input_links={len(self.input_links)})"

    def copyWith(
        self,
        input_links: Optional[list[Link]] = None,
        bias: Optional[float] = None,
        activation: Optional[Activation] = None,
    ):
        return Neuron(
            input_links=self.input_links if input_links is None else input_links,
            bias=self.bias if bias is None else bias,
            activation=self.activation if activation is None else activation,
        )
