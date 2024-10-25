import numpy as np
import wandb

from learning_annealers import *
from utils import *
from activation import *
from optimizers import *
from losses import *


class NeuralNetwork:
    def __init__(self,
                 layer_info: list = [784, 10],
                 num_hidden_layers: int = 2,
                 num_node_per_hidden_layer: int = 32,
                 weight_decay: float = 0.005,
                 learning_rate: float = 1e-4,
                 optimizer: str = 'adam',
                 batch_size: int = 16,
                 weights_init: str = 'xavier_normal',
                 activation: str = 'relu',
                 beta: float = 0.9,
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 epsilon: float = 0.00000001,
                 loss: str = 'cross_entropy',
                 save_model=False
                 ) -> None:
        '''
        :param layer_info: <list>, optional, Default: [784, 10]
            List specifying the number of nodes in each layer, including the input and output layers.
            Example: [784, 128, 64, 10] represents a network with 784 input nodes, two hidden layers with 128 and 64 nodes, and 10 output nodes.
        :param num_hidden_layers: <int>, optional, Default: 2
            Number of hidden layers in the neural network.
        :param num_node_per_hidden_layer: <int>, optional, Default: 32
            Number of nodes per hidden layer in the neural network.
        :param weight_decay: <float>, optional, Default: 0.005
            Coefficient for L2 regularization to prevent overfitting.
        :param learning_rate: <float>, optional, Default: 1e-3
            Learning rate for gradient descent optimization during training.
        :param optimizer: <str>, optional, Default: 'sgd'
            Optimization algorithm to use. Supported values: 'sgd' (Stochastic Gradient Descent), 'adam' (Adam optimization).
        :param batch_size: <int>, optional, Default: 64
            Number of samples per mini-batch during training.
        :param weights_init: <str>, optional, Default: 'random'
            Method for weight initialization. Supported values: 'random', 'xavier', 'he'.
        :param activation: <str>, optional, Default: 'relu'
            Activation function to use for hidden layers. Supported values: 'relu', 'sigmoid', 'tanh'.
        :param beta: <float>, Default: 0.9, 
            The beta parameter for optimizers
        :param beta1: <float>, Default: 0.9, 
            The beta parameter for optimizers
        :param beta2: <float>, Default: 0.999, 
            The beta parameter for optimizers
        :param loss: <str>, Default: cross_entropy
            Type of loss that you want to chose: mse, cross_entropy.
        :param save_model: <bool> optional, Default: True
            For saving the model with best weights
        '''

        self.input = layer_info[0]
        self.output = layer_info[1]

        self.l2_regularization = weight_decay
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.weights_init = weights_init
        self.activation = activation
        self.beta = beta
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.loss = loss
        self.save_model = save_model

        if type(num_node_per_hidden_layer) == int:
            num_node_per_hidden_layer = [
                num_node_per_hidden_layer]*num_hidden_layers

        self.layer_info = [
            self.input] + [num_node_per_hidden_layer[i] for i in range(num_hidden_layers)]+[self.output]
        self.variables = self._init_weights(weights_init)

    def _init_weights(self, type: str = None) -> dict:
        '''
        _init_weights() Initialiaze weights function

        :param type: <str> Type of weight initialization
        :return: <dict> A dictionary containing all the layer weights and biases
        '''
        self.type = type
        self.variables = {}
        for ind, (dim0, dim1) in enumerate(zip(self.layer_info[:-1], self.layer_info[1:])):
            if self.type == 'random':
                self.variables['w_b_' + str(ind)] = (
                    np.random.normal(size=(dim0, dim1)),
                    np.random.normal(size=(1, dim1))
                )
            elif self.type == 'xavier_normal':
                std_dev = np.sqrt(2.0 / (dim0 + dim1))
                self.variables['w_b_' + str(ind)] = (
                    np.random.normal(0, std_dev, size=(dim0, dim1)),
                    np.random.normal(size=(1, dim1))
                )
            elif self.type == 'xavier_uniform':
                limit = np.sqrt(6.0 / (dim0 + dim1))
                self.variables['w_b_' + str(ind)] = (
                    np.random.uniform(-limit, limit, size=(dim0, dim1)),
                    np.random.normal(size=(1, dim1))
                )
            elif self.type == 'he_normal':
                std_dev = np.sqrt(2.0 / dim0)
                self.variables['w_b_' + str(ind)] = (
                    np.random.normal(0, std_dev, size=(dim0, dim1)),
                    np.random.normal(size=(1, dim1))
                )
            elif self.type == 'he_uniform':
                limit = np.sqrt(6.0 / dim0)
                self.variables['w_b_' + str(ind)] = (
                    np.random.uniform(-limit, limit, size=(dim0, dim1)),
                    np.random.normal(size=(1, dim1))
                )

        self.opt_parameters = {}
        if self.optimizer != 'sgd':
            for ind, (dim0, dim1) in enumerate(zip(self.layer_info[:-1], self.layer_info[1:])):
                self.opt_parameters['velocity_dw' +
                                    str(ind)] = np.zeros((dim0, dim1))
                self.opt_parameters['velocity_db' +
                                    str(ind)] = np.zeros((1, dim1))
                self.opt_parameters['square_dw' +
                                    str(ind)] = np.zeros((dim0, dim1))
                self.opt_parameters['square_db' +
                                    str(ind)] = np.zeros((1, dim1))
                if self.optimizer == 'nesterov' or self.optimizer == 'nadam':
                    self.opt_parameters['velocity_curr_dw' +
                                        str(ind)] = np.zeros((dim0, dim1))
                    self.opt_parameters['velocity_curr_db' +
                                        str(ind)] = np.zeros((1, dim1))
                    self.opt_parameters['square_curr_dw' +
                                        str(ind)] = np.zeros((dim0, dim1))
                    self.opt_parameters['square_curr_db' +
                                        str(ind)] = np.zeros((1, dim1))

        return self.variables

    def forward(self, data: np.ndarray, weights_and_biases: dict) -> np.ndarray:
        '''
        forward() Forward propagation function

        :param data: <list|np.ndarray> Data is a list of data or numpy array taking as input
        :param weights_and_biases: <dict> A dictionary containing all the layer weights and biases
        :return:<dict>,<dict> Array of output returned after forward propagation,Cached Results
        '''
        keys = list(weights_and_biases.keys())
        cache_results = {}

        A = data
        for ind in range(len(keys)):
            A_prev = A
            weight = weights_and_biases[keys[ind]][0]
            bias = weights_and_biases[keys[ind]][1]
            Z = np.dot(A_prev, weight) + bias
            if ind < len(keys) - 1:
                if self.activation == 'relu':
                    A = Activation.relu(Z)
                elif self.activation == 'tanh':
                    A = Activation.tanh(Z)
                elif self.activation == 'sigmoid':
                    A = Activation.sigmoid(Z)
            else:
                A = Activation.softmax(Z)
            cache_results['A_Z_'+str(ind)] = (A_prev, Z)

        return A, cache_results

    def backward(self, pred: np.ndarray, labels: np.ndarray, weights_and_biases: dict, cache_results: dict) -> dict:
        '''
        backward() Backward Function

        :param pred: <list|np.ndarray> prediction values of the data
        :param labels: <list|np.ndarray> True labels of the data
        :param weights_and_biases: <dict> A dictionary containing all the layer weights and biases
        :return: <dict> A dictionary containing all the layer weights and biases gradients
        '''
        batch_size = len(pred)
        grads = {}

        caches_keys = list(cache_results.keys())
        wb_keys = list(weights_and_biases.keys())
        num_caches = len(caches_keys)

        A_prev, Z = cache_results[caches_keys[num_caches-1]]
        if self.loss == 'cross_entropy':
            dZ_last = (pred - labels)
        elif self.loss == 'squared_error':
            dZ_last = (pred - labels)*pred - pred @ \
                (np.dot((pred - labels).T, pred))

        grads['GradW_'+str(num_caches)] = np.dot(A_prev.T,
                                                 dZ_last)/batch_size
        grads['GradB_'+str(num_caches)] = np.sum(dZ_last,
                                                 axis=0, keepdims=True)/batch_size
        grads['dA_'+str(num_caches-1)] = np.dot(
            weights_and_biases['w_b_'+str(num_caches-1)][0], dZ_last.T)

        for i in range(num_caches-2, -1, -1):
            A_prev, Z = cache_results[caches_keys[i]]
            dA_prev = grads['dA_'+str(i+1)].T
            if self.activation == 'relu':
                dZ = np.multiply(dA_prev, Activation.relu(Z, derivative=True))
            if self.activation == 'tanh':
                dZ = np.multiply(dA_prev, Activation.tanh(Z, derivative=True))
            if self.activation == 'sigmoid':
                dZ = np.multiply(
                    dA_prev, Activation.sigmoid(Z, derivative=True))

            grads['GradW_'+str(i+1)] = np.dot(A_prev.T, dZ)/batch_size + \
                self.l2_regularization*weights_and_biases['w_b_'+str(i)][0]
            grads['GradB_'+str(i+1)] = np.sum(dZ, axis=0,
                                              keepdims=True)/batch_size
            grads['dA_'+str(i)] = np.dot(
                weights_and_biases['w_b_'+str(i)][0], dZ.T)

            if self.activation == 'relu':
                grads['GradW_'+str(i+1)] = np.clip(grads['GradW_' +
                                                         str(i+1)], a_min=-1, a_max=1)
                grads['GradB_'+str(i+1)] = np.clip(grads['GradB_' +
                                                         str(i+1)], a_min=-1, a_max=1)
                grads['dA_'+str(i)] = np.clip(grads['dA_' +
                                                    str(i)], a_min=-1, a_max=1)

        return grads

    def train(self, train: tuple, validation: tuple, test: tuple, epochs: int = 100, log: bool = True) -> None:
        '''
        train() Function to start training

        :param train: <tuple> Tuple containing train data and train labels
        :param test: <tuple> Tuple containing test data and test labels
        :param epochs: <int> Number of epochs
        :param log: <bool> Whether to log
        '''
        batch_size = self.batch_size
        callback = BestModelCallback(filepath='best_model.pkl')

        if self.loss == 'cross_entropy':
            loss = cross_entropy_loss
        elif self.loss == 'squared_error':
            loss = squared_error

        for _ in range(1, epochs+1):
            train_data, train_labels = train
            train_batches = np.arange(len(train_data)//batch_size)
            np.random.shuffle(train_batches)

            train_loss = 0
            train_acc = 0

            for i, ind in enumerate(train_batches):
                batch_data = train_data[ind*batch_size:(ind+1)*batch_size]
                batch_labels = train_labels[ind*batch_size:(ind+1)*batch_size]
                probs, cache_results = self.forward(batch_data, self.variables)
                train_loss += loss(probs, batch_labels)
                train_acc += accuracy(probs, batch_labels)

                if self.optimizer == 'sgd':
                    grads = self.backward(
                        probs, batch_labels, self.variables, cache_results)
                    self.variables = Optimizers.sgd(
                        self.variables, grads, self.learning_rate)
                elif self.optimizer == 'momentum':
                    grads = self.backward(
                        probs, batch_labels, self.variables, cache_results)
                    self.variables, self.opt_parameters = Optimizers.momentum(
                        self.variables, grads, self.learning_rate, beta=self.beta, opt_parameters=self.opt_parameters)
                elif self.optimizer == 'nesterov':
                    self.opt_parameters['grad_func'] = self.backward
                    self.opt_parameters['forward_func'] = self.forward
                    self.opt_parameters['data'] = batch_data
                    self.opt_parameters['labels'] = batch_labels
                    self.variables, self.opt_parameters = Optimizers.nesterov(
                        self.variables, None, self.learning_rate, beta=self.beta, opt_parameters=self.opt_parameters)
                elif self.optimizer == 'adagrad':
                    grads = self.backward(
                        probs, batch_labels, self.variables, cache_results)
                    self.variables, self.opt_parameters = Optimizers.adagrad(
                        self.variables, grads, self.learning_rate, opt_parameters=self.opt_parameters)
                elif self.optimizer == 'rmsprop':
                    grads = self.backward(
                        probs, batch_labels, self.variables, cache_results)
                    self.variables, self.opt_parameters = Optimizers.rmsprop(
                        self.variables, grads, self.learning_rate, beta=self.beta, opt_parameters=self.opt_parameters)
                elif self.optimizer == 'adadelta':
                    grads = self.backward(
                        probs, batch_labels, self.variables, cache_results)
                    self.variables, self.opt_parameters = Optimizers.adadelta(
                        self.variables, grads, self.learning_rate, beta=self.beta, opt_parameters=self.opt_parameters)
                elif self.optimizer == 'adam':
                    grads = self.backward(
                        probs, batch_labels, self.variables, cache_results)
                    self.variables, self.opt_parameters = Optimizers.adam(
                        self.variables, grads, self.learning_rate, iter_number=_, beta=self.beta1, beta2=self.beta2, opt_parameters=self.opt_parameters)
                elif self.optimizer == 'maxprop':
                    grads = self.backward(
                        probs, batch_labels, self.variables, cache_results)
                    self.variables, self.opt_parameters = Optimizers.maxprop(
                        self.variables, grads, self.learning_rate, beta=self.beta, opt_parameters=self.opt_parameters)
                elif self.optimizer == 'adamax':
                    grads = self.backward(
                        probs, batch_labels, self.variables, cache_results)
                    self.variables, self.opt_parameters = Optimizers.adamax(
                        self.variables, grads, self.learning_rate, iter_number=_, beta=self.beta1, beta2=self.beta2, opt_parameters=self.opt_parameters)
                elif self.optimizer == 'nadam':
                    self.opt_parameters['grad_func'] = self.backward
                    self.opt_parameters['forward_func'] = self.forward
                    self.opt_parameters['data'] = batch_data
                    self.opt_parameters['labels'] = batch_labels
                    self.variables, self.opt_parameters = Optimizers.nadam(
                        self.variables, None, self.learning_rate, iter_number=_, beta=self.beta1, beta2=self.beta2, opt_parameters=self.opt_parameters)

            if log:
                wandb.log({'train_loss': train_loss/i})
                wandb.log({'train_accuracy': train_acc/i})

            if (_ % 5 == 0) and (not log):
                print(
                    '[{0}/{1}]\t train_loss:{2:.3f}\t train_acc: {3:.3f}'.format('0'*(len(str(epochs))-len(str(_)))+str(_), epochs, train_loss/i, train_acc/i), end='\t')

            validation_data, validation_labels = validation
            validation_batches = np.arange(len(validation_data)//batch_size)
            np.random.shuffle(train_batches)

            validation_loss = 0
            validation_acc = 0
            predictions = np.array([])
            labels = np.array([])

            for i, ind in enumerate(validation_batches):
                batch_data = validation_data[ind*batch_size:(ind+1)*batch_size]
                batch_labels = validation_labels[ind *
                                                 batch_size:(ind+1)*batch_size]
                probs, cache_results = self.forward(batch_data, self.variables)
                validation_loss += loss(probs, batch_labels)
                validation_acc += accuracy(probs, batch_labels)
                predictions = np.concatenate(
                    (predictions, probs.argmax(1)), axis=0)
                labels = np.concatenate(
                    (labels, batch_labels.argmax(1)), axis=0)

            if log:
                wandb.log({'validation_loss': validation_loss/i})
                wandb.log({'validation_accuracy': validation_acc/i})
                class_names = ["t-shirt_top", 'trouser_pants', 'pullover shirt',
                               'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
                wandb.log({"conf_mat": wandb.plot.confusion_matrix(
                    probs=None, y_true=labels, preds=predictions, class_names=class_names)})

            if (_ % 5 == 0) and (not log):
                print(
                    'validation_loss:{0:.3f}\t validation_acc: {1:.3f}'.format(validation_loss/i, validation_acc/i))

            test_data, test_labels = test
            test_batches = np.arange(len(test_data)//batch_size)
            np.random.shuffle(train_batches)

            test_loss = 0
            test_acc = 0
            predictions = np.array([])
            labels = np.array([])

            for i, ind in enumerate(test_batches):
                batch_data = test_data[ind*batch_size:(ind+1)*batch_size]
                batch_labels = test_labels[ind*batch_size:(ind+1)*batch_size]

                probs, cache_results = self.forward(batch_data, self.variables)
                test_loss += loss(probs, batch_labels)
                test_acc += accuracy(probs, batch_labels)
                predictions = np.concatenate(
                    (predictions, probs.argmax(1)), axis=0)
                labels = np.concatenate(
                    (labels, batch_labels.argmax(1)), axis=0)

            if self.save_model:
                callback(model_weights=self.variables, current_loss=test_loss)

            if log:
                wandb.log({'test_loss': test_loss/i})
                wandb.log({'test_accuracy': test_acc/i})
                class_names = ["t-shirt_top", 'trouser_pants', 'pullover shirt',
                               'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
                wandb.log({"conf_mat": wandb.plot.confusion_matrix(
                    probs=None, y_true=labels, preds=predictions, class_names=class_names)})

            if (_ % 5 == 0) and (not log):
                print(
                    'test_loss:{0:.3f}\t test_acc: {1:.3f}'.format(test_loss/i, test_acc/i))
