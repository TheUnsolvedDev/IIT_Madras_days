import numpy as np


class Activation:
    '''
    Activation Function class to contain all sorts of the activation function
    for building a neural network
    '''

    @staticmethod
    def relu(x: np.ndarray, derivative: bool = False) -> np.ndarray:
        '''
        relu() ReLU(Rectified Linear Unit) Function

        :param x: <int|float|np.ndarray> Input array for using relu function
        :param derivative: <bool>, Default: False, Boolean flag to return relu value or relu derivatve
        :return: <int|float|np.ndarray> Result after applying relu function
        '''
        if derivative:
            x = np.where(x < 0, 0, x)
            x = np.where(x >= 0, 1, x)
            return x
        return np.maximum(0, x)

    @staticmethod
    def sigmoid(x: np.ndarray, derivative: bool = False) -> np.ndarray:
        '''
        sigmoid() Sigmoid Function

        :param x: <int|float|np.ndarray> Input array for using sigmoid function
        :param derivative: <bool>, Default: False, Boolean flag to return sigmoid value or sigmoid derivatve
        :return: <int|float|np.ndarray> Result after applying sigmoid function
        '''
        if derivative:
            return (np.exp(-x))/((np.exp(-x)+1)**2)
        return 1/(1 + np.exp(-x))

    @staticmethod
    def tanh(x: np.ndarray, derivative: bool = False) -> np.ndarray:
        '''
        tanh() Tanh Function

        :param x: <int|float|np.ndarray> Input array for using tanh function
        :param derivative: <bool>, Default: False, Boolean flag to return tanh value or tanh derivatve
        :return: <int|float|np.ndarray> Result after applying tanh function
        '''
        if derivative:
            return 1 - (np.tanh(x)**2)
        return np.tanh(x)

    @staticmethod
    def selu(x: np.ndarray, derivative: bool = False) -> np.ndarray:
        '''
        selu() Scaled Exponential Linear Unit (SELU) Function

        :param x: <int|float|np.ndarray> Input array for using selu function
        :param derivative: <bool>, Default: False, Boolean flag to return selu value or selu derivatve
        :return: <int|float|np.ndarray> Result after applying selu function
        '''
        alpha = 1.6732
        scale = 1.0507
        if derivative:
            return scale * np.where(x > 0, 1, alpha * np.exp(x))
        return scale * np.where(x > 0, x, alpha * (np.exp(x) - 1))

    @staticmethod
    def gelu(x: np.ndarray, derivative: bool = False) -> np.ndarray:
        '''
        gelu() Gaussian Error Linear Unit (GELU) Function

        :param x: <int|float|np.ndarray> Input array for using gelu function
        :param derivative: <bool>, Default: False, Boolean flag to return gelu value or gelu derivatve
        :return: <int|float|np.ndarray> Result after applying gelu function
        '''
        if derivative:
            return 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))

    @staticmethod
    def leaky_relu(x: np.ndarray, alpha: float = 0.01, derivative: bool = False) -> np.ndarray:
        '''
        leaky_relu() Leaky ReLU Function

        :param x: <int|float|np.ndarray> Input array for using leaky_relu function
        :param alpha: <float>, Default: 0.01, Slope of the negative part of the function
        :param derivative: <bool>, Default: False, Boolean flag to return leaky_relu value or leaky_relu derivatve
        :return: <int|float|np.ndarray> Result after applying leaky_relu function
        '''
        if derivative:
            x = np.where(x > 0, 1, alpha)
            return x
        return np.where(x > 0, x, alpha * x)

    @staticmethod
    def elu(x: np.ndarray, alpha: float = 1.0, derivative: bool = False) -> np.ndarray:
        '''
        elu() Exponential Linear Unit (ELU) Function

        :param x: <int|float|np.ndarray> Input array for using elu function
        :param alpha: <float>, Default: 1.0, The alpha value for the function
        :param derivative: <bool>, Default: False, Boolean flag to return elu value or elu derivatve
        :return: <int|float|np.ndarray> Result after applying elu function
        '''
        if derivative:
            return np.where(x > 0, 1, alpha * np.exp(x))
        return np.where(x > 0, x, alpha * (np.exp(x) - 1))

    @staticmethod
    def swish(x: np.ndarray, derivative: bool = False) -> np.ndarray:
        '''
        swish() Swish Function

        :param x: <int|float|np.ndarray> Input array for using swish function
        :param derivative: <bool>, Default: False, Boolean flag to return swish value or swish derivatve
        :return: <int|float|np.ndarray> Result after applying swish function
        '''
        if derivative:
            return Activation.sigmoid(x) + x * Activation.sigmoid(x, derivative=True) * (1 - Activation.sigmoid(x))
        return x * Activation.sigmoid(x)

    @staticmethod
    def softplus(x: np.ndarray, derivative: bool = False) -> np.ndarray:
        '''
        softplus() Softplus Function

        :param x: <int|float|np.ndarray> Input array for using softplus function
        :param derivative: <bool>, Default: False, Boolean flag to return softplus value or softplus derivatve
        :return: <int|float|np.ndarray> Result after applying softplus function
        '''
        if derivative:
            return 1 / (1 + np.exp(-x))
        return np.log(1 + np.exp(x))

    @staticmethod
    def mish(x: np.ndarray, derivative: bool = False) -> np.ndarray:
        '''
        mish() Mish Function

        :param x: <int|float|np.ndarray> Input array for using mish function
        :param derivative: <bool>, Default: False, Boolean flag to return mish value or mish derivatve
        :return: <int|float|np.ndarray> Result after applying mish function
        '''
        if derivative:
            return (np.exp(x) * (4 * x + 4) + 4 * np.exp(2 * x) + np.exp(3 * x) + np.exp(x) * (4 * x - 4) + 4 * np.exp(2 * x) * x) / (np.exp(2 * x) + 2 * np.exp(x) + 2) ** 2
        return x * np.tanh(np.log(1 + np.exp(x)))

    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        '''
        softmax() Softmax Function

        :param x: <int|float|np.ndarray> Input array for using softmax function
        :return: <int|float|np.ndarray> Result after applying softmax function
        '''

        x_max = np.amax(np.asarray_chkfinite(x), axis=1, keepdims=True)
        exps = np.exp(x - x_max)
        return exps / np.sum(exps, axis=1, keepdims=True)
