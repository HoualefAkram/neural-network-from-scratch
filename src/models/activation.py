from typing import Callable
from math import log, exp


class Activation:
    def __init__(self, func: Callable[[float], float]):
        self.func: Callable[[float], float] = func

    def call(self, x):
        if self.func:
            return self.func(x)
        return x


class SoftPlus(Activation):
    def __func(self, x):
        return log(1 + exp(x))

    def __init__(self):
        super().__init__(func=self.__func)


class ReLU(Activation):
    def __func(self, x):
        return max(0, x)

    def __init__(self):
        super().__init__(func=self.__func)
