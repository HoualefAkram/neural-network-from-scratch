class Error:

    @staticmethod
    def calc(predicted_outputs: list[float], outputs: list[float]) -> float:
        raise NotImplementedError

    @staticmethod
    def de_dp(predicted_outputs: list[float], outputs: list[float]):
        raise NotImplementedError


class Mse(Error):

    @staticmethod
    def calc(predicted_outputs: list[list[float]], outputs: list[list[float]]) -> float:
        if len(predicted_outputs) != len(outputs):
            raise ValueError("Input lists must have the same length")

        n = len(predicted_outputs)
        if n == 0:
            raise ValueError("Input lists must not be empty")

        total = 0.0
        for p_list, o_list in zip(predicted_outputs, outputs):
            if len(p_list) != len(o_list):
                raise ValueError("predicted/output values must be the same length")
            total += sum((p_val - o_val) ** 2 for p_val, o_val in zip(p_list, o_list))
        return total / n
