import numpy as np
from constants import STUDY_SPEED, ITERATION_NUMBER, ITERATION_INFO, ALLOWABLE_MISTAKE


class Sigmoid:

    def __call__(self, number):
        return number

    def derivative(self, number):
        return 1


class Relu(Sigmoid):

    def __call__(self, number):
        if number > 0:
            return number
        return 0

    def derivative(self, number):
        if number > 0:
            return 1
        return 0


class Task(Sigmoid):

    def __call__(self, number: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-number))

    def derivative(self, number):
        sgm = self(number)
        return sgm * (1 - sgm)


class SimpleAI:
    def __init__(self, input_number: int = 3, neurons_number: int = 4, sigmoid: Sigmoid = Task()):
        self.input_number = input_number
        self.wages = neurons_number
        self.activate = sigmoid

    @property
    def activate(self):
        return self._activate

    @activate.setter
    def activate(self, sigmoid):
        if isinstance(sigmoid, Sigmoid):
            self._activate = sigmoid
        else:
            raise TypeError("Error, enter no sigmoid class")
    @property
    def input_number(self):
        return self._input_number

    @input_number.setter
    def input_number(self, input_number):
        if isinstance(input_number, int):
            if input_number > 0:
                self._input_number = input_number
            else:
                raise ValueError("Input number can`t be less then 1")
        else:
            raise TypeError("Non int input number")

    @property
    def wages(self):
        return self._wages

    @wages.setter
    def wages(self, neurons_number):
        if not isinstance(neurons_number, int):
            raise TypeError("Neurons number must be a int")
        self._wages = {}
        self._wages["hidden_w"] = np.random.rand(self.input_number, neurons_number)
        self._wages["hidden_b"] = np.random.rand(1, neurons_number)
        self._wages["out_w"] = np.random.rand(neurons_number, 1)
        self._wages["out_b"] = np.random.rand(1, 1)

        self._wages["hidden_w"] = (self._wages["hidden_w"] - 0.5) * 2 * np.sqrt(1 / self.input_number)
        self._wages["hidden_b"] = (self._wages["hidden_b"] - 0.5) * 2 * np.sqrt(1 / self.input_number)
        self._wages["out_w"] = (self._wages["out_w"] - 0.5) * 2 * np.sqrt(1 / neurons_number)
        self._wages["out_b"] = (self._wages["out_b"] - 0.5) * 2 * np.sqrt(1 / neurons_number)

    def forward_propagation(self, input_data: np.ndarray):
        result = {}
        result["hidden_s"] = input_data @ self._wages["hidden_w"] + self.wages["hidden_b"]
        result["hidden_y"] = self.activate(result["hidden_s"])
        result["out"] = result["hidden_y"] @ self.wages["out_w"] + self.wages["out_b"]
        return result

    def back_propagation(self, input: np.ndarray, output: np.ndarray, result):
        delta_wages = {}
        delta_2 = result["out"] - output
        delta_wages["delta_out_w"] = result["hidden_y"].T @ delta_2
        delta_wages["delta_out_b"] = np.sum(delta_2, axis=0, keepdims=True)
        delta_h1 = delta_2 @ self.wages["out_w"].T
        delta_1 = delta_h1 * self.activate.derivative(result["hidden_s"])
        delta_wages["delta_hidden_w"] = input.T @ delta_1
        delta_wages["delta_hidden_b"] = np.sum(delta_1, axis=0, keepdims=True)

        return delta_wages


    def update_wages(self, delta_wages):
        self._wages["hidden_w"] -= STUDY_SPEED*delta_wages["delta_hidden_w"]
        self._wages["hidden_b"] -= STUDY_SPEED*delta_wages["delta_hidden_b"]
        self._wages["out_w"] -= STUDY_SPEED*delta_wages["delta_out_w"]
        self._wages["out_b"] -= STUDY_SPEED*delta_wages["delta_out_b"]

    def study(self, input_data: np.ndarray, output_data: np.ndarray):
        for i in range(ITERATION_NUMBER):
            result = self.forward_propagation(input_data)



            self.update_wages(self.back_propagation(input_data, output_data, result))

            if i % ITERATION_INFO == 0:
                total_error = np.sum((result["out"] - output_data) ** 2)
                print(str(i) + " iterations MISTAKE:" + str(total_error))
