class Error:
    def calc(self, predicted_outputs: list[float], outputs: list[float]) -> float:
        raise NotImplementedError


class MSE(Error):
    def calc(self, predicted_outputs: list[float], outputs: list[float]) -> float:
        if len(predicted_outputs) != len(outputs):
            raise ValueError("Input lists must have the same length")

        n = len(predicted_outputs)
        if n == 0:
            raise ValueError("Input lists must not be empty")

        return sum((p - o) ** 2 for p, o in zip(predicted_outputs, outputs)) / n
