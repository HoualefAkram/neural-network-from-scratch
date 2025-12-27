from .activation import Activation
from .link import Link
from typing import Optional


class Neuron:

    def __init__(
        self,
        bias: float,
        activation: Activation,
        value: Optional[float] = None,
        input_links: Optional[list[Link]] = [],
    ):
        self.value = value
        self.bias: float = bias
        self.activation: Activation = activation
        self.input_links: list[Link] = input_links

    def __repr__(self):
        return f"Neuron(bias={self.bias},activation={type(self.activation)}, input_links={len(self.input_links)})"

    def copyWith(
        self,
        value: Optional[float] = None,
        input_links: Optional[list[Link]] = None,
        bias: Optional[float] = None,
        activation: Optional[Activation] = None,
    ):
        return Neuron(
            value=self.value if (value is None) else value,
            input_links=self.input_links if (input_links is None) else input_links,
            bias=self.bias if (bias is None) else bias,
            activation=self.activation if (activation is None) else activation,
        )
