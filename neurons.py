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
    def __init__(self, input_number: int = 3, neurons_number: int = 4,
                 sigmoid: Sigmoid = Relu()):
        self.activate = sigmoid
        self.input_number = input_number
        self.wages = neurons_number

    @property
    def activate(self):
        return self._activate

    @activate.setter
    def activate(self, sigmoid):
        if isinstance(sigmoid, Sigmoid):
            self._activate = lambda t: sigmoid.result(t)
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
        self._wages = [6*np.random.random((neurons_number, self.input_number)) - 3]
        self._wages += [6*np.random.random((1, neurons_number)) - 3]

    def forward_propagation(self, input_data: np.ndarray):
        if not isinstance(input_data, np.ndarray):
            raise TypeError("Input data must be a numpy array(ndarray)")
        if input_data.size != self.input_number:
            raise ValueError("Size of input data array must be length as input number")
        result = input_data
        for i in self.wages:
            result = np.array([self.activate(k) for k in np.dot(i, result)])
        return result

    def back_propagation(self, delta):
        print(delta)

    def study(self, input_data: np.ndarray, result_data: np.ndarray):
        if not isinstance(result_data, np.ndarray) or not isinstance(input_data, np.ndarray):
            raise TypeError("Input and result data must be a numpy array(ndarray)")
        if input_data.size != result_data.size:
            raise ValueError("Sizes of input and result data must be the same")

        print(input_data)


x = SimpleAI(neurons_number=4)
print(x.wages)
print(x.forward_propagation(np.array([1, 1, 1])))
