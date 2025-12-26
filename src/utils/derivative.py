from typing import Callable


class Derivative:
    EPSILON: float = 1e-10

    @staticmethod
    def dydx(func: Callable[[float], float], point: float):
        return (func(point + Derivative.EPSILON) - func(point)) / Derivative.EPSILON
