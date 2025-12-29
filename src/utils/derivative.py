from typing import Callable


class Derivative:
    EPSILON: float = 1e-6

    @staticmethod
    def dydx(func: Callable[[float], float], point: float):
        return (func(point + Derivative.EPSILON) - func(point - Derivative.EPSILON)) / (
            2 * Derivative.EPSILON
        )
