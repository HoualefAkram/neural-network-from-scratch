from math import sqrt
from random import gauss


class WeightGenerator:

    @staticmethod
    def he_weight(layer_len):
        sigma: float = sqrt(2 / layer_len)
        mu: float = 0
        random_weight: float = gauss(mu=mu, sigma=sigma)
        print(random_weight)
        return random_weight
