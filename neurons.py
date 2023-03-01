import numpy as np


class Sigmoid:
    @staticmethod
    def result(number):
        return number


class Relu(Sigmoid):
    @staticmethod
    def result(number):
        if number > 0:
            return number
        return 0


class SimpleAI:
    def __init__(self, input_number: int = 3, slices_number: int = 1, neurons_number: np.ndarray = np.array([4]),
                 sigmoid: Sigmoid = Relu()):
        self.sigmoid = sigmoid
        self.input_number = input_number
        self.slices_number = slices_number
        self.wages = neurons_number

    @property
    def sigmoid(self):
        return self._sigmoid

    @sigmoid.setter
    def sigmoid(self, sigmoid):
        if isinstance(sigmoid, Sigmoid):
            self._sigmoid = sigmoid
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
    def slices_number(self):
        return self._slices_number

    @slices_number.setter
    def slices_number(self, slices_number):
        if isinstance(slices_number, int):
            if slices_number >= 0:
                self._slices_number = slices_number
            else:
                raise ValueError("Slices number can`t be less then 0")
        else:
            raise TypeError("Non int slices number")

    @property
    def wages(self):
        return self._wages

    @wages.setter
    def wages(self, neurons_number):
        if not isinstance(neurons_number, np.ndarray):
            raise TypeError("Neurons number must be a numpy array(ndarray)")
        if not self.slices_number:
            self._wages = 2*np.random.random((1, self.input_number)) - 1
        else:
            if self.slices_number > 


x = SimpleAI(slices_number=1)
print(x.wages)
