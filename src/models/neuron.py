from .activation import Activation
from .link import Link
from typing import Optional


class Neuron:

    def __init__(
        self,
        id: int,
        bias: float,
        activation: Activation,
        value: Optional[float] = None,
        input_links: Optional[list[Link]] = None,
    ):
        self.value = value
        self.id = id
        self.bias: float = bias
        self.activation: Activation = activation
        self.input_links: list[Link] = input_links

    def __repr__(self):
        return f"Neuron(id={self.id},bias={self.bias},activation={type(self.activation)}, input_links={len(self.input_links) if self.input_links is not None else "None"})"

    def copyWith(
        self,
        id: Optional[int] = None,
        value: Optional[float] = None,
        input_links: Optional[list[Link]] = None,
        bias: Optional[float] = None,
        activation: Optional[Activation] = None,
    ):
        return Neuron(
            id=self.id if (id is None) else id,
            value=self.value if (value is None) else value,
            input_links=self.input_links if (input_links is None) else input_links,
            bias=self.bias if (bias is None) else bias,
            activation=self.activation if (activation is None) else activation,
        )

    def get_value(self):
        # needed if the input is the value of the neuron
        if self.value is not None:
            return self.value

        z = (
            sum(il.weight * il.source.get_value() for il in self.input_links)
            + self.bias
        )
        return self.activation(z)


class NeuronProxy:
    _counter: int = 0

    @classmethod
    def create(
        cls,
        bias: float,
        activation: Activation,
        value: Optional[float] = None,
        input_links: Optional[list[Link]] = None,
    ) -> Neuron:
        neuron_id = cls._counter
        cls._counter += 1

        return Neuron(
            id=neuron_id,
            bias=bias,
            activation=activation,
            value=value,
            input_links=input_links,
        )

    @classmethod
    def reset_counter(cls):
        cls._counter = 0
