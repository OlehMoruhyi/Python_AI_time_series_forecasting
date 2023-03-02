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
            self._sigmoid = lambda t: sigmoid.result(t)
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
            self._wages = [6*np.random.random((1, self.input_number)) - 3]
        else:
            if self.slices_number > neurons_number.size:
                raise ValueError("Too short list of neurons number")
            else:
                self._wages = [6*np.random.random((neurons_number[0], self.input_number)) - 3]
                for i in range(1, self.slices_number):
                    self._wages += [6*np.random.random((neurons_number[i], neurons_number[i-1])) - 3]
                self._wages += [6*np.random.random((1, neurons_number[self.slices_number-1])) - 3]

    def forecasting(self, input_data: np.ndarray):
        if not isinstance(input_data, np.ndarray):
            raise TypeError("Input data must be a numpy array(ndarray)")
        if input_data.size != self.input_number:
            raise ValueError("Size of input data array must be length as input number")
        result = input_data
        for i in self.wages:
            result = np.array([self.sigmoid(k) for k in np.dot(i, result)])
        return result

    def study(self, input_data: np.ndarray, result_data: np.ndarray):
        if not isinstance(result_data, np.ndarray) or not isinstance(input_data, np.ndarray):
            raise TypeError("Input and result data must be a numpy array(ndarray)")
        if input_data.size != result_data.size:
            raise ValueError("Sizes of input and result data must be the same")
        print(input_data)


x = SimpleAI(slices_number=1, neurons_number=np.array([4, 3]))
print(x.wages)
print(x.forecasting(np.array([1, 1, 1])))
