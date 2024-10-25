import numpy as np
import time


class Optimizers:
    '''
    Optimzer class for conataining all the algorithms for storing all sorts of optimizers
    '''

    @staticmethod
    def sgd(weights_and_bias: dict, grads: dict, alpha: float) -> dict:
        '''
        sgd() Stochastic Gradient Descent Algorithm

        :param weights and bias: <dict{}> Weights and bias of the neural network
        :param grads: <dict{}> Gradients of the weights and bias of the Neural Network
        :param alpha: <float> Learning rate value
        :return : <dict{}> Updated weights and biases
        '''
        grad_keys = list(grads.keys())
        wb_keys = list(weights_and_bias.keys())
        weights = [weights_and_bias[key][0] for key in wb_keys]
        biases = [weights_and_bias[key][1] for key in wb_keys]

        grad_weights = [grads[grad_keys[i]]
                        for i in range(0, len(grad_keys), 3)][::-1]
        grad_biases = [grads[grad_keys[i+1]]
                       for i in range(0, len(grad_keys), 3)][::-1]

        for i in range(len(weights)):
            w, dw = weights[i], grad_weights[i]
            w -= alpha*dw
            b, db = biases[i], grad_biases[i]
            b -= alpha*db
            weights_and_bias[wb_keys[i]] = (w, b)
        return weights_and_bias

    @staticmethod
    def momentum(weights_and_bias: dict, grads: dict, alpha: float, beta: float, opt_parameters: dict = {}, Lambda=None) -> dict:
        '''
        momentum() Momentum based gradient descent algorithm

        :param weights and bias: <dict{}> Weights and bias of the neural network
        :param grads: <dict{}> Gradients of the weights and bias of the Neural Network
        :param alpha: <float> Learning rate value
        :param beta: <float> Beta parameter of the velocity
        :param opt_parameters: <dict{}> optional parameters for the algorithm
        :return : <dict{}> Updated weights and biases
        '''
        grad_keys = list(grads.keys())
        wb_keys = list(weights_and_bias.keys())
        weights = [weights_and_bias[key][0] for key in wb_keys]
        biases = [weights_and_bias[key][1] for key in wb_keys]

        grad_weights = [grads[grad_keys[i]]
                        for i in range(0, len(grad_keys), 3)][::-1]
        grad_biases = [grads[grad_keys[i+1]]
                       for i in range(0, len(grad_keys), 3)][::-1]

        for l in range(len(weights)):
            w, dw = weights[l], grad_weights[l]
            b, db = biases[l], grad_biases[l]
            opt_parameters['velocity_dw' +
                           str(l)] = beta*opt_parameters['velocity_dw' + str(l)] + dw
            opt_parameters['velocity_db' +
                           str(l)] = beta*opt_parameters['velocity_db' + str(l)] + db
            w -= alpha*opt_parameters['velocity_dw'+str(l)]
            b -= alpha*opt_parameters['velocity_db'+str(l)]
            weights_and_bias[wb_keys[l]] = (w, b)
        return weights_and_bias, opt_parameters

    @staticmethod
    def nesterov(weights_and_bias: dict, grads: dict, alpha: float, beta: float, opt_parameters: dict = {}, Lambda=None) -> dict:
        '''
        nesterov() Nesterov based accelerated gradient descent algorithm

        :param weights and bias: <dict{}> Weights and bias of the neural network
        :param grads: <dict{}> Gradients of the weights and bias of the Neural Network
        :param alpha: <float> Learning rate value
        :param beta: <float> Beta parameter of the velocity
        :param opt_parameters: <dict{}> optional parameters for the algorithm
        :return : <dict{}> Updated weights and biases
        '''
        wb_keys = list(weights_and_bias.keys())
        weights = [weights_and_bias[key][0] for key in wb_keys]
        biases = [weights_and_bias[key][1] for key in wb_keys]

        temp_weights_and_bias = {}
        for l in range(len(weights)):
            w = weights[l]
            b = biases[l]
            w_temp = w - beta*opt_parameters['velocity_dw'+str(l)]
            b_temp = b - beta*opt_parameters['velocity_db'+str(l)]
            temp_weights_and_bias[wb_keys[l]] = (w_temp, b_temp)

        forward_results, cache_results = opt_parameters['forward_func'](
            opt_parameters['data'], temp_weights_and_bias)
        grads = opt_parameters['grad_func'](
            forward_results, opt_parameters['labels'], temp_weights_and_bias, cache_results)
        grad_keys = list(grads.keys())
        grad_weights = [grads[grad_keys[i]]
                        for i in range(0, len(grad_keys), 3)][::-1]
        grad_biases = [grads[grad_keys[i+1]]
                       for i in range(0, len(grad_keys), 3)][::-1]

        for l in range(len(weights)):
            w, dw = weights[l], grad_weights[l]
            b, db = biases[l], grad_biases[l]

            w -= beta * opt_parameters['velocity_dw'+str(l)] + alpha*dw
            b -= beta * opt_parameters['velocity_db'+str(l)] + alpha*db

            opt_parameters['velocity_dw'+str(l)] = beta * \
                opt_parameters['velocity_dw'+str(l)] + alpha*dw
            opt_parameters['velocity_db'+str(l)] = beta * \
                opt_parameters['velocity_db'+str(l)] + alpha*db
            weights_and_bias[wb_keys[l]] = (w, b)
        return weights_and_bias, opt_parameters

    @staticmethod
    def adagrad(weights_and_bias: dict, grads: dict, alpha: float, opt_parameters: dict = {}, Lambda=None) -> dict:
        '''
        adagrad() adaptive gradient based gradient descent algorithm

        :param weights and bias: <dict{}> Weights and bias of the neural network
        :param grads: <dict{}> Gradients of the weights and bias of the Neural Network
        :param alpha: <float> Learning rate value
        :param beta: <float> Beta parameter of the velocity
        :param opt_parameters: <dict{}> optional parameters for the algorithm
        :return : <dict{}> Updated weights and biases
        '''
        grad_keys = list(grads.keys())
        wb_keys = list(weights_and_bias.keys())
        weights = [weights_and_bias[key][0] for key in wb_keys]
        biases = [weights_and_bias[key][1] for key in wb_keys]
        epsilon = 1e-8

        grad_weights = [grads[grad_keys[i]]
                        for i in range(0, len(grad_keys), 3)][::-1]
        grad_biases = [grads[grad_keys[i+1]]
                       for i in range(0, len(grad_keys), 3)][::-1]

        for l in range(len(weights)):
            w, dw = weights[l], grad_weights[l]
            b, db = biases[l], grad_biases[l]
            opt_parameters['velocity_dw' +
                           str(l)] = opt_parameters['velocity_dw' + str(l)] + dw**2
            opt_parameters['velocity_db' +
                           str(l)] = opt_parameters['velocity_db' + str(l)] + db**2
            w -= alpha * \
                (1/np.sqrt(opt_parameters['velocity_dw' +
                 str(l)]+epsilon))*dw
            b -= alpha * \
                (1/np.sqrt(opt_parameters['velocity_db' +
                 str(l)]+epsilon))*db
            weights_and_bias[wb_keys[l]] = (w, b)
        return weights_and_bias, opt_parameters

    @staticmethod
    def rmsprop(weights_and_bias: dict, grads: dict, alpha: float, beta: float, opt_parameters: dict = {}) -> dict:
        '''
        rmsprop() RMSprop based gradient descent algorithm

        :param weights and bias: <dict{}> Weights and bias of the neural network
        :param grads: <dict{}> Gradients of the weights and bias of the Neural Network
        :param alpha: <float> Learning rate value
        :param beta: <float> Beta2 parameter of the velocity
        :param opt_parameters: <dict{}> optional parameters for the algorithm
        :return : <dict{}> Updated weights and biases
        '''
        grad_keys = list(grads.keys())
        wb_keys = list(weights_and_bias.keys())
        weights = [weights_and_bias[key][0] for key in wb_keys]
        biases = [weights_and_bias[key][1] for key in wb_keys]

        grad_weights = [grads[grad_keys[i]]
                        for i in range(0, len(grad_keys), 3)][::-1]
        grad_biases = [grads[grad_keys[i+1]]
                       for i in range(0, len(grad_keys), 3)][::-1]
        epsilon = 1e-4

        for l in range(len(weights)):
            w, dw = weights[l], grad_weights[l]
            b, db = biases[l], grad_biases[l]
            opt_parameters['square_dw'+str(l)] = beta*opt_parameters['square_dw' +
                                                                     str(l)] + (1-beta)*np.square(dw)
            opt_parameters['square_db'+str(l)] = beta*opt_parameters['square_db' +
                                                                     str(l)] + (1-beta)*np.square(db)
            w -= (alpha * dw) / \
                (np.sqrt(opt_parameters['square_dw' + str(l)])+epsilon)
            b -= (alpha * db) / \
                (np.sqrt(opt_parameters['square_db' + str(l)])+epsilon)
            weights_and_bias[wb_keys[l]] = (w, b)
        return weights_and_bias, opt_parameters

    @staticmethod
    def adadelta(weights_and_bias: dict, grads: dict, alpha: float, beta: float, opt_parameters: dict = {}) -> dict:
        '''
        adadelta() adadelta based gradient descent algorithm

        :param weights and bias: <dict{}> Weights and bias of the neural network
        :param grads: <dict{}> Gradients of the weights and bias of the Neural Network
        :param alpha: <float> Learning rate value
        :param beta: <float> Beta2 parameter of the velocity
        :param opt_parameters: <dict{}> optional parameters for the algorithm
        :return : <dict{}> Updated weights and biases
        '''
        grad_keys = list(grads.keys())
        wb_keys = list(weights_and_bias.keys())
        weights = [weights_and_bias[key][0] for key in wb_keys]
        biases = [weights_and_bias[key][1] for key in wb_keys]

        grad_weights = [grads[grad_keys[i]]
                        for i in range(0, len(grad_keys), 3)][::-1]
        grad_biases = [grads[grad_keys[i+1]]
                       for i in range(0, len(grad_keys), 3)][::-1]
        epsilon = 1e-5

        for l in range(len(weights)):
            w, dw = weights[l], grad_weights[l]
            b, db = biases[l], grad_biases[l]
            opt_parameters['square_dw'+str(l)] = beta*opt_parameters['square_dw' +
                                                                     str(l)] + (1-beta)*np.square(dw)
            opt_parameters['square_db'+str(l)] = beta*opt_parameters['square_db' +
                                                                     str(l)] + (1-beta)*np.square(db)

            del_w = -(np.sqrt(opt_parameters['velocity_dw'+str(l)] + epsilon)/np.sqrt(
                opt_parameters['square_dw'+str(l)]+epsilon))*dw
            del_b = -(np.sqrt(opt_parameters['velocity_db'+str(l)] + epsilon)/np.sqrt(
                opt_parameters['square_db'+str(l)]+epsilon))*db

            w += del_w
            b += del_b

            opt_parameters['velocity_dw'+str(l)] = beta*opt_parameters['velocity_dw' +
                                                                       str(l)] + (1-beta)*np.square(del_w)
            opt_parameters['velocity_db'+str(l)] = beta*opt_parameters['velocity_db' +
                                                                       str(l)] + (1-beta)*np.square(del_b)

            weights_and_bias[wb_keys[l]] = (w, b)
        return weights_and_bias, opt_parameters

    @staticmethod
    def adam(weights_and_bias: dict, grads: dict, alpha: float, iter_number: int, beta: float, beta2: float, opt_parameters={}):
        '''
        adam() Adam based gradient descent algorithm

        :param weights and bias: <dict{}> Weights and bias of the neural network
        :param grads: <dict{}> Gradients of the weights and bias of the Neural Network
        :param alpha: <float> Learning rate value
        :param beta: <float> Beta parameter of the velocity
        :param beta2: <float> Beta2 parameter of the velocity
        :param opt_parameters: <dict{}> optional parameters for the algorithm
        :return : <dict{}> Updated weights and biases
        '''
        grad_keys = list(grads.keys())
        wb_keys = list(weights_and_bias.keys())
        weights = [weights_and_bias[key][0] for key in wb_keys]
        biases = [weights_and_bias[key][1] for key in wb_keys]

        grad_weights = [grads[grad_keys[i]]
                        for i in range(0, len(grad_keys), 3)][::-1]
        grad_biases = [grads[grad_keys[i+1]]
                       for i in range(0, len(grad_keys), 3)][::-1]

        for l in range(len(weights)):
            w, dw = weights[l], grad_weights[l]
            b, db = biases[l], grad_biases[l]
            opt_parameters['velocity_dw'+str(l)] = beta*opt_parameters['velocity_dw' +
                                                                       str(l)] + (1-beta)*dw
            opt_parameters['velocity_db'+str(l)] = beta*opt_parameters['velocity_db' +
                                                                       str(l)] + (1-beta)*db
            opt_parameters['square_dw'+str(l)] = beta2*opt_parameters['square_dw' +
                                                                      str(l)] + (1-beta2)*np.square(dw)
            opt_parameters['square_db'+str(l)] = beta2*opt_parameters['square_db' +
                                                                      str(l)] + (1-beta2)*np.square(db)
            velocity_dw = opt_parameters['velocity_dw' +
                                         str(l)] / (1-(beta**iter_number))
            velocity_db = opt_parameters['velocity_db' +
                                         str(l)] / (1-(beta**iter_number))
            square_dw = opt_parameters['square_dw' +
                                       str(l)] / (1-(beta2**iter_number))
            square_db = opt_parameters['square_db' +
                                       str(l)] / (1-(beta2**iter_number))
            w -= alpha * (velocity_dw / (np.sqrt(square_dw)+10**-8))
            b -= alpha * (velocity_db / (np.sqrt(square_db)+10**-8))
            weights_and_bias[wb_keys[l]] = (w, b)
        return weights_and_bias, opt_parameters

    @staticmethod
    def maxprop(weights_and_bias: dict, grads: dict, alpha: float, beta: float, opt_parameters: dict = {}) -> dict:
        '''
        maxprop() MAXprop based gradient descent algorithm

        :param weights and bias: <dict{}> Weights and bias of the neural network
        :param grads: <dict{}> Gradients of the weights and bias of the Neural Network
        :param alpha: <float> Learning rate value
        :param beta: <float> Beta2 parameter of the velocity
        :param opt_parameters: <dict{}> optional parameters for the algorithm
        :return : <dict{}> Updated weights and biases
        '''
        grad_keys = list(grads.keys())
        wb_keys = list(weights_and_bias.keys())
        weights = [weights_and_bias[key][0] for key in wb_keys]
        biases = [weights_and_bias[key][1] for key in wb_keys]

        grad_weights = [grads[grad_keys[i]]
                        for i in range(0, len(grad_keys), 3)][::-1]
        grad_biases = [grads[grad_keys[i+1]]
                       for i in range(0, len(grad_keys), 3)][::-1]
        epsilon = 1e-8

        for l in range(len(weights)):
            w, dw = weights[l], grad_weights[l]
            b, db = biases[l], grad_biases[l]
            opt_parameters['square_dw'+str(l)] = np.maximum(beta *
                                                            opt_parameters['square_dw' + str(l)], np.abs(dw))
            opt_parameters['square_db'+str(l)] = np.maximum(beta *
                                                            opt_parameters['square_db' + str(l)], np.abs(db))
            w -= (alpha * dw) / \
                (np.sqrt(opt_parameters['square_dw' + str(l)])+epsilon)
            b -= (alpha * db) / \
                (np.sqrt(opt_parameters['square_db' + str(l)])+epsilon)
            weights_and_bias[wb_keys[l]] = (w, b)
        return weights_and_bias, opt_parameters

    @staticmethod
    def adamax(weights_and_bias: dict, grads: dict, alpha: float, iter_number: int, beta: float, beta2: float, opt_parameters={}):
        '''
        adamax() Adamax based gradient descent algorithm

        :param weights and bias: <dict{}> Weights and bias of the neural network
        :param grads: <dict{}> Gradients of the weights and bias of the Neural Network
        :param alpha: <float> Learning rate value
        :param beta: <float> Beta parameter of the velocity
        :param beta2: <float> Beta2 parameter of the velocity
        :param opt_parameters: <dict{}> optional parameters for the algorithm
        :return : <dict{}> Updated weights and biases
        '''
        grad_keys = list(grads.keys())
        wb_keys = list(weights_and_bias.keys())
        weights = [weights_and_bias[key][0] for key in wb_keys]
        biases = [weights_and_bias[key][1] for key in wb_keys]

        grad_weights = [grads[grad_keys[i]]
                        for i in range(0, len(grad_keys), 3)][::-1]
        grad_biases = [grads[grad_keys[i+1]]
                       for i in range(0, len(grad_keys), 3)][::-1]
        epsilon = 1e-8

        for l in range(len(weights)):
            w, dw = weights[l], grad_weights[l]
            b, db = biases[l], grad_biases[l]
            opt_parameters['velocity_dw'+str(l)] = beta*opt_parameters['velocity_dw' +
                                                                       str(l)] + (1-beta)*dw
            opt_parameters['velocity_db'+str(l)] = beta*opt_parameters['velocity_db' +
                                                                       str(l)] + (1-beta)*db
            velocity_dw = opt_parameters['velocity_dw' +
                                         str(l)] / (1-(beta**iter_number))
            velocity_db = opt_parameters['velocity_db' +
                                         str(l)] / (1-(beta**iter_number))
            opt_parameters['square_dw'+str(l)] = np.maximum(beta2 *
                                                            opt_parameters['square_dw' + str(l)], np.abs(dw))
            opt_parameters['square_db'+str(l)] = np.maximum(beta2 *
                                                            opt_parameters['square_db' + str(l)], np.abs(db))

            w -= (alpha*velocity_dw)/(opt_parameters['square_dw'+str(l)]+epsilon)
            b -= (alpha*velocity_db)/(opt_parameters['square_db'+str(l)]+epsilon)

            weights_and_bias[wb_keys[l]] = (w, b)
        return weights_and_bias, opt_parameters

    @staticmethod
    def nadam(weights_and_bias: dict, grads: dict, alpha: float, iter_number: int, beta: float, beta2: float, opt_parameters={}):
        '''
        nadam() Nesterov Adam based gradient descent algorithm

        :param weights and bias: <dict{}> Weights and bias of the neural network
        :param grads: <dict{}> Gradients of the weights and bias of the Neural Network
        :param alpha: <float> Learning rate value
        :param beta: <float> Beta parameter of the velocity
        :param beta2: <float> Beta2 parameter of the velocity
        :param opt_parameters: <dict{}> optional parameters for the algorithm
        :return : <dict{}> Updated weights and biases
        '''
        wb_keys = list(weights_and_bias.keys())
        weights = [weights_and_bias[key][0] for key in wb_keys]
        biases = [weights_and_bias[key][1] for key in wb_keys]
        epsilon = 1e-8

        temp_weights_and_bias = {}
        for l in range(len(weights)):
            w = weights[l]
            b = biases[l]
            velocity_dw = opt_parameters['velocity_dw' +
                                         str(l)] / (1-(beta**iter_number))
            velocity_db = opt_parameters['velocity_db' +
                                         str(l)] / (1-(beta**iter_number))
            square_dw = opt_parameters['square_dw' +
                                       str(l)] / (1-(beta2**iter_number))
            square_db = opt_parameters['square_db' +
                                       str(l)] / (1-(beta2**iter_number))
            w_temp = w - beta * (velocity_dw / (np.sqrt(square_dw)+10**-8))
            b_temp = b - beta * (velocity_db / (np.sqrt(square_db)+10**-8))
            temp_weights_and_bias[wb_keys[l]] = (w_temp, b_temp)

        forward_results, cache_results = opt_parameters['forward_func'](
            opt_parameters['data'], temp_weights_and_bias)
        grads = opt_parameters['grad_func'](
            forward_results, opt_parameters['labels'], temp_weights_and_bias, cache_results)
        grad_keys = list(grads.keys())
        grad_weights = [grads[grad_keys[i]]
                        for i in range(0, len(grad_keys), 3)][::-1]
        grad_biases = [grads[grad_keys[i+1]]
                       for i in range(0, len(grad_keys), 3)][::-1]

        grad_weights = [grads[grad_keys[i]]
                        for i in range(0, len(grad_keys), 3)][::-1]
        grad_biases = [grads[grad_keys[i+1]]
                       for i in range(0, len(grad_keys), 3)][::-1]

        for l in range(len(weights)):
            w, dw = weights[l], grad_weights[l]
            b, db = biases[l], grad_biases[l]
            opt_parameters['velocity_dw'+str(l)] = beta*opt_parameters['velocity_dw' +
                                                                       str(l)] + alpha*dw
            opt_parameters['velocity_db'+str(l)] = beta*opt_parameters['velocity_db' +
                                                                       str(l)] + alpha*db
            opt_parameters['square_dw'+str(l)] = beta2*opt_parameters['square_dw' +
                                                                      str(l)] + (1-beta2)*np.square(dw)
            opt_parameters['square_db'+str(l)] = beta2*opt_parameters['square_db' +
                                                                      str(l)] + (1-beta2)*np.square(db)
            velocity_dw = opt_parameters['velocity_dw' +
                                         str(l)] / (1-(beta**iter_number))
            velocity_db = opt_parameters['velocity_db' +
                                         str(l)] / (1-(beta**iter_number))
            square_dw = opt_parameters['square_dw' +
                                       str(l)] / (1-(beta2**iter_number))
            square_db = opt_parameters['square_db' +
                                       str(l)] / (1-(beta2**iter_number))

            w -= (alpha/np.sqrt(square_dw+epsilon)) * \
                (beta*velocity_dw + ((1-beta)*dw)/(1-(beta**iter_number)))
            b -= (alpha/np.sqrt(square_db+epsilon)) * \
                (beta*velocity_db + ((1-beta)*db)/(1-(beta**iter_number)))

            weights_and_bias[wb_keys[l]] = (w, b)
        return weights_and_bias, opt_parameters
