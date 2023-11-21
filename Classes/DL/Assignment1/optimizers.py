import numpy as np

class Optimizers:
    @staticmethod
    def sgd(weights_and_bias, grads, alpha):
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
    def momentum(weights_and_bias, grads, alpha, beta, opt_parameters={}, Lambda=None):
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
                                                                       str(l)] + (1-beta)*grad_weights[l]
            opt_parameters['velocity_db'+str(l)] = beta*opt_parameters['velocity_db' +
                                                                       str(l)] + (1-beta)*grad_biases[l]
            w -= alpha*opt_parameters['velocity_dw'+str(l)]
            b -= alpha*opt_parameters['velocity_db'+str(l)]
            weights_and_bias[wb_keys[l]] = (w, b)
        return weights_and_bias, opt_parameters

    @staticmethod
    def nesterov(weights, grads):
        return weights

    @staticmethod
    def rmsprop(weights_and_bias, grads, alpha, beta2, opt_parameters={}, Lambda=None):
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
            opt_parameters['square_dw'+str(l)] = beta2*opt_parameters['square_dw' +
                                                                      str(l)] + (1-beta2)*np.square(grad_weights[l])
            opt_parameters['square_db'+str(l)] = beta2*opt_parameters['square_db' +
                                                                      str(l)] + (1-beta2)*np.square(grad_biases[l])
            w -= alpha * \
                (grad_weights[l] /
                 (np.sqrt(opt_parameters['square_dw'+str(l)])+10**-8))
            b -= alpha * \
                (grad_biases[l] /
                 (np.sqrt(opt_parameters['square_db'+str(l)])+10**-8))
            weights_and_bias[wb_keys[l]] = (w, b)
        return weights_and_bias, opt_parameters

    @staticmethod
    def adam(weights_and_bias, grads, alpha, iter_number, beta, beta2, opt_parameters={}, Lambda=None):
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
                                                                       str(l)] + (1-beta)*grad_weights[l]
            opt_parameters['velocity_db'+str(l)] = beta*opt_parameters['velocity_db' +
                                                                       str(l)] + (1-beta)*grad_biases[l]
            opt_parameters['square_dw'+str(l)] = beta2*opt_parameters['square_dw' +
                                                                      str(l)] + (1-beta2)*np.square(grad_weights[l])
            opt_parameters['square_db'+str(l)] = beta2*opt_parameters['square_db' +
                                                                      str(l)] + (1-beta2)*np.square(grad_biases[l])
            velocity_dw = opt_parameters['velocity_dw' +
                                         str(l)] / (1-(beta**iter_number))
            velocity_db = opt_parameters['velocity_db' +
                                         str(l)] / (1-(beta**iter_number))
            square_dw = opt_parameters['square_dw' +
                                       str(l)] / (1-(beta**iter_number))
            square_db = opt_parameters['square_db' +
                                       str(l)] / (1-(beta**iter_number))
            w -= alpha * (velocity_dw / (np.sqrt(square_dw)+10**-8))
            b -= alpha * (velocity_db / (np.sqrt(square_db)+10**-8))
            weights_and_bias[wb_keys[l]] = (w, b)
        return weights_and_bias, opt_parameters

    @staticmethod
    def nadam(weights, grads):
        return weights