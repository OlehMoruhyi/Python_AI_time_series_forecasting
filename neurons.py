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
        self._wages["w1"] = np.random.rand(self.input_number, neurons_number)
        self._wages["b1"] = np.random.rand(1, neurons_number)
        self._wages["w2"] = np.random.rand(neurons_number, 1)
        self._wages["b2"] = np.random.rand(1, 1)

        self._wages["w1"] = (self._wages["w1"] - 0.5) * 2 * np.sqrt(1 / self.input_number)
        self._wages["b1"] = (self._wages["b1"] - 0.5) * 2 * np.sqrt(1 / self.input_number)
        self._wages["w2"] = (self._wages["w2"] - 0.5) * 2 * np.sqrt(1 / neurons_number)
        self._wages["b2"] = (self._wages["b2"] - 0.5) * 2 * np.sqrt(1 / neurons_number)

    def forward_propagation(self, input_data: np.ndarray):
        result = {}
        result["t1"] = input_data @ self._wages["w1"] + self.wages["b1"]
        result["h1"] = self.activate(result["t1"])
        result["z"] = result["h1"] @ self.wages["w2"] + self.wages["b2"]
        return result

    def back_propagation(self, input: np.ndarray, output: np.ndarray, result):
        delta_wages = {}
        delta_2 = result["z"] - output
        delta_wages["dw2"] = result["h1"].T @ delta_2
        delta_wages["db2"] = np.sum(delta_2, axis=0, keepdims=True)
        delta_h1 = delta_2 @ self.wages["w2"].T
        delta_1 = delta_h1 * self.activate.derivative(result["t1"])
        delta_wages["dw1"] = input.T @ delta_1
        delta_wages["db1"] = np.sum(delta_1, axis=0, keepdims=True)

        return delta_wages


    def update_wages(self, delta_wages):
        self._wages["w1"] -= STUDY_SPEED*delta_wages["dw1"]
        self._wages["b1"] -= STUDY_SPEED*delta_wages["db1"]
        self._wages["w2"] -= STUDY_SPEED*delta_wages["dw2"]
        self._wages["b2"] -= STUDY_SPEED*delta_wages["db2"]

    def study(self, input_data: np.ndarray, output_data: np.ndarray):
        total_error0 = 0
        for i in range(ITERATION_NUMBER):
            result = self.forward_propagation(input_data)



            self.update_wages(self.back_propagation(input_data, output_data, result))

            if i % ITERATION_INFO == 0:
                total_error = np.sum((result["z"] - output_data) ** 2)
                print(str(i) + " iterations MISTAKE:" + str(total_error))
                total_error0 = total_error

