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
                                                                       str(l)] + (1-beta)*dw
            opt_parameters['velocity_db'+str(l)] = beta*opt_parameters['velocity_db' +
                                                                       str(l)] + (1-beta)*db
            w -= alpha*opt_parameters['velocity_dw'+str(l)]
            b -= alpha*opt_parameters['velocity_db'+str(l)]
            weights_and_bias[wb_keys[l]] = (w, b)
        return weights_and_bias, opt_parameters

    @staticmethod
    def nesterov(weights_and_bias, grads, alpha, beta, opt_parameters={}, Lambda=None):
        wb_keys = list(weights_and_bias.keys())
        weights = [weights_and_bias[key][0] for key in wb_keys]
        biases = [weights_and_bias[key][1] for key in wb_keys]

        temp_weights_and_bias = {}
        for l in range(len(weights)):
            w = weights[l]
            b = biases[l]
            w_temp = w - alpha*opt_parameters['velocity_dw'+str(l)]
            b_temp = b - alpha*opt_parameters['velocity_db'+str(l)]
            temp_weights_and_bias[wb_keys[l]] = (w_temp, b_temp)

        forward_results = opt_parameters['forward_func'](
            opt_parameters['data'], temp_weights_and_bias)
        grads = opt_parameters['grad_func'](
            forward_results, opt_parameters['labels'], temp_weights_and_bias)
        grad_keys = list(grads.keys())
        grad_weights = [grads[grad_keys[i]]
                        for i in range(0, len(grad_keys), 3)][::-1]
        grad_biases = [grads[grad_keys[i+1]]
                       for i in range(0, len(grad_keys), 3)][::-1]

        for l in range(len(weights)):
            w, dw = weights[l], grad_weights[l]
            b, db = biases[l], grad_biases[l]

            opt_parameters['velocity_curr_dw'+str(l)] = beta * \
                opt_parameters['velocity_dw'+str(l)] + \
                (1-beta)*dw
            opt_parameters['velocity_curr_db'+str(l)] = beta * \
                opt_parameters['velocity_db'+str(l)] + \
                (1-beta)*db
            w -= alpha * opt_parameters['velocity_curr_dw'+str(l)]
            b -= alpha * opt_parameters['velocity_curr_db'+str(l)]
            weights_and_bias[wb_keys[l]] = (w, b)
        return weights_and_bias, opt_parameters

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
                                                                      str(l)] + (1-beta2)*np.square(dw)
            opt_parameters['square_db'+str(l)] = beta2*opt_parameters['square_db' +
                                                                      str(l)] + (1-beta2)*np.square(db)
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
    def nadam(weights_and_bias, grads, alpha, iter_number, beta, beta2, opt_parameters={}, Lambda=None):
        wb_keys = list(weights_and_bias.keys())
        weights = [weights_and_bias[key][0] for key in wb_keys]
        biases = [weights_and_bias[key][1] for key in wb_keys]

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
            w_temp = w - alpha * (velocity_dw / (np.sqrt(square_dw)+10**-8))
            b_temp = b - alpha * (velocity_db / (np.sqrt(square_db)+10**-8))
            temp_weights_and_bias[wb_keys[l]] = (w_temp, b_temp)

        forward_results = opt_parameters['forward_func'](
            opt_parameters['data'], temp_weights_and_bias)
        grads = opt_parameters['grad_func'](
            forward_results, opt_parameters['labels'], temp_weights_and_bias)
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
            opt_parameters['velocity_curr_dw'+str(l)] = beta*opt_parameters['velocity_dw' +
                                                                            str(l)] + (1-beta)*dw
            opt_parameters['velocity_curr_db'+str(l)] = beta*opt_parameters['velocity_db' +
                                                                            str(l)] + (1-beta)*db
            opt_parameters['square_curr_dw'+str(l)] = beta2*opt_parameters['square_dw' +
                                                                           str(l)] + (1-beta2)*np.square(dw)
            opt_parameters['square_curr_db'+str(l)] = beta2*opt_parameters['square_db' +
                                                                           str(l)] + (1-beta2)*np.square(db)
            velocity_dw = opt_parameters['velocity_curr_dw' +
                                         str(l)] / (1-(beta**iter_number))
            velocity_db = opt_parameters['velocity_curr_db' +
                                         str(l)] / (1-(beta**iter_number))
            square_dw = opt_parameters['square_curr_dw' +
                                       str(l)] / (1-(beta2**iter_number))
            square_db = opt_parameters['square_curr_db' +
                                       str(l)] / (1-(beta2**iter_number))
            w -= alpha * (velocity_dw / (np.sqrt(square_dw)+10**-8))
            b -= alpha * (velocity_db / (np.sqrt(square_db)+10**-8))
            weights_and_bias[wb_keys[l]] = (w, b)
        return weights_and_bias, opt_parameters
