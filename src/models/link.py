from neuron import Neuron


class Link:

    def __init__(self, weight, parent, child):
        self.weight: float = weight
        self.parent: Neuron = parent
        self.child: Neuron = child
