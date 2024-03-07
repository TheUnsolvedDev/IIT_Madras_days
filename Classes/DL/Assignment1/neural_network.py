import numpy as np
import matplotlib.pyplot as plt
from activations import *
from optimizers import *
from losses import *
import wandb


def one_hot(a, num_classes):
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)])


def accuracy(probs, labels):
    return np.mean(probs.argmax(axis=1) == labels.argmax(axis=1))


class NeuralNetwork:
    def __init__(self, layer_info=[784, 10], num_hidden=2, num_nodes=[64, 32], weight_decay=0.005,  learning_rate=0.001, optimizer='nadam', batch_size=256, weights_init='xavier_normal', activation='tanh') -> None:
        self.input = layer_info[0]
        self.output = layer_info[-1]

        self.l2_regularization = weight_decay
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.weights_init = weights_init
        self.activation = activation

        if type(num_nodes) == int:
            num_nodes = [num_nodes]*num_hidden

        self.layer_info = [self.input] + \
            [num_nodes[i] for i in range(num_hidden)]+[self.output]
        self.variables = self._init_weights(weights_init)
        self.cache_results = {}

    def _init_weights(self, type=None):
        variables = {}
        if type == 'random':
            for ind, (dim0, dim1) in enumerate(zip(self.layer_info[:-1], self.layer_info[1:])):
                variables['w_b_' + str(ind)] = (np.random.normal(
                    size=(dim0, dim1)), np.random.normal(size=(1, dim1)))
        elif type == 'xavier_normal':
            for ind, (dim0, dim1) in enumerate(zip(self.layer_info[:-1], self.layer_info[1:])):
                variables['w_b_' + str(ind)] = (np.random.normal(0, 2.0/(dim0+dim1),
                                                                 size=(dim0, dim1)), np.random.normal(size=(1, dim1)))
        elif type == 'xavier_uniform':
            for ind, (dim0, dim1) in enumerate(zip(self.layer_info[:-1], self.layer_info[1:])):
                variables['w_b_' + str(ind)] = (np.random.normal(-(np.sqrt(6.0/(dim0+dim1))), (np.sqrt(6.0/(dim0+dim1))),
                                                                 size=(dim0, dim1)), np.random.normal(size=(1, dim1)))
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

        return variables

    def forward(self, data, weights_and_biases):
        keys = list(weights_and_biases.keys())

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
            self.cache_results['A_Z_'+str(ind)] = (A_prev, Z)
        return A

    def backward(self, pred, labels, weights_and_biases):
        batch_size = len(pred)
        grads = {}

        caches_keys = list(self.cache_results.keys())
        num_caches = len(caches_keys)

        A_prev, Z = self.cache_results[caches_keys[num_caches-1]]
        dZ_last = pred - labels

        grads['dW_'+str(num_caches)] = np.dot(A_prev.T,
                                              dZ_last)/batch_size
        grads['db_'+str(num_caches)] = np.sum(dZ_last,
                                              axis=0, keepdims=True)/batch_size
        grads['dA_'+str(num_caches-1)] = np.dot(
            weights_and_biases['w_b_'+str(num_caches-1)][0], dZ_last.T)

        for i in range(num_caches-2, -1, -1):
            A_prev, Z = self.cache_results[caches_keys[i]]
            dA_prev = grads['dA_'+str(i+1)].T
            if self.activation == 'relu':
                dZ = np.multiply(dA_prev, Activation.relu(Z, derivative=True))
            if self.activation == 'tanh':
                dZ = np.multiply(dA_prev, Activation.tanh(Z, derivative=True))
            if self.activation == 'sigmoid':
                dZ = np.multiply(
                    dA_prev, Activation.sigmoid(Z, derivative=True))

            grads['dW_'+str(i+1)] = np.dot(A_prev.T, dZ)/batch_size + \
                self.l2_regularization*weights_and_biases['w_b_'+str(i)][0]
            grads['db_'+str(i+1)] = np.sum(dZ, axis=0,
                                           keepdims=True)/batch_size
            grads['dA_'+str(i)] = np.dot(
                weights_and_biases['w_b_'+str(i)][0], dZ.T)

            if self.activation == 'relu':
                grads['dW_'+str(i+1)] = np.clip(grads['dW_' +
                                                      str(i+1)], a_min=-1, a_max=1)
                grads['db_'+str(i+1)] = np.clip(grads['db_' +
                                                      str(i+1)], a_min=-1, a_max=1)
                grads['dA_'+str(i)] = np.clip(grads['dA_' +
                                                    str(i)], a_min=-1, a_max=1)

        return grads

    def train_one_step(self, weights_and_biases, probs, labels, iter_number=None, opt_parameters=None):
        if self.optimizer == 'sgd':
            grads = self.backward(probs, labels, weights_and_biases)
            return Optimizers.sgd(weights_and_biases, grads, self.learning_rate)
        elif self.optimizer == 'momentum':
            grads = self.backward(probs, labels, weights_and_biases)
            return Optimizers.momentum(weights_and_biases, grads, self.learning_rate, beta=0.9, opt_parameters=opt_parameters)
        elif self.optimizer == 'nesterov':
            opt_parameters['grad_func'] = self.backward
            opt_parameters['data'] = self.forward
            opt_parameters['prediction'] = probs
            opt_parameters['labels'] = labels
            return Optimizers.nesterov(weights_and_biases, None, self.learning_rate, beta=0.9, opt_parameters=opt_parameters)
        elif self.optimizer == 'rmsprop':
            grads = self.backward(probs, labels, weights_and_biases)
            return Optimizers.rmsprop(weights_and_biases, grads, self.learning_rate, beta2=0.999, opt_parameters=opt_parameters)
        elif self.optimizer == 'adam':
            grads = self.backward(probs, labels, weights_and_biases)
            return Optimizers.adam(weights_and_biases, grads, self.learning_rate, iter_number=iter_number, beta=0.9, beta2=0.999, opt_parameters=opt_parameters)
        elif self.optimizer == 'nadam':
            opt_parameters['grad_func'] = self.backward
            opt_parameters['forward_func'] = self.forward
            opt_parameters['data'] = probs
            opt_parameters['labels'] = labels
            return Optimizers.nadam(weights_and_biases, None, self.learning_rate, iter_number=iter_number, beta=0.9, beta2=0.999, opt_parameters=opt_parameters)

    def train(self, train, test, epochs=100, log=True):
        batch_size = self.batch_size
        train_losses, test_losses = [], []
        train_accs, test_accs = [], []

        for _ in range(1, epochs+1):
            train_data, train_labels = train
            train_batches = np.arange(len(train_data)//batch_size)
            np.random.shuffle(train_batches)
            train_loss = 0
            train_acc = 0
            for i, ind in enumerate(train_batches):
                batch_data = train_data[ind*batch_size:(ind+1)*batch_size]
                batch_labels = train_labels[ind*batch_size:(ind+1)*batch_size]

                probs = self.forward(batch_data, self.variables)
                train_loss += cross_entropy_loss(probs, batch_labels)
                train_acc += accuracy(probs, batch_labels)

                if self.optimizer == 'sgd':
                    self.variables = self.train_one_step(
                        self.variables, probs, batch_labels)
                elif self.optimizer == 'momentum':
                    self.variables, self.opt_parameters = self.train_one_step(
                        self.variables, probs, batch_labels, _, self.opt_parameters)
                elif self.optimizer == 'nesterov':
                    self.variables, self.opt_parameters = self.train_one_step(
                        self.variables, batch_data, batch_labels, _, self.opt_parameters)
                elif self.optimizer == 'rmsprop':
                    self.variables, self.opt_parameters = self.train_one_step(
                        self.variables, probs, batch_labels, _, self.opt_parameters)
                elif self.optimizer == 'adam':
                    self.variables, self.opt_parameters = self.train_one_step(
                        self.variables, probs, batch_labels, _, self.opt_parameters)
                elif self.optimizer == 'nadam':
                    self.variables, self.opt_parameters = self.train_one_step(
                        self.variables, batch_data, batch_labels, _, self.opt_parameters)

            if (_ % 5 == 0) and (not log):
                print(
                    '[{0}/{1}]\t train_loss:{2:.3f}\t train_acc: {3:.3f}'.format('0'*(len(str(epochs))-len(str(_)))+str(_), epochs, train_loss/i, train_acc/i), end='\t')
            if log:
                wandb.log({'epoch': _})
                wandb.log({'train_loss': train_loss/i})
                wandb.log({'train_accuracy': train_acc/i})

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

                probs = self.forward(batch_data, self.variables)
                test_loss += cross_entropy_loss(probs, batch_labels)
                test_acc += accuracy(probs, batch_labels)
                predictions = np.concatenate(
                    (predictions, probs.argmax(1)), axis=0)
                labels = np.concatenate(
                    (labels, batch_labels.argmax(1)), axis=0)

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


sweep_config = {
    'method': 'bayes',
}

metric = {
    'name': 'test_accuracy',
    'goal': 'maximize'
}
sweep_config['metric'] = metric
parameters_dict = {
    'number_of_epochs': {
        'values': [5, 10]
    },
    'number_of_hidden_layers': {
        'values': [3, 4, 5]
    },
    'size_of_every_hidden_layer': {
        'values': [32, 64, 128]
    },
    'weight_decay': {
        'values': [0, 0.0005, 0.5]
    },
    'learning_rate': {
        'values': [1e-3, 1e-4]
    },
    'optimizer': {
        'values': ['sgd', 'momentum', 'nesterov', 'rmsprop', 'adam', 'nadam']
    },
    'batch_size': {
        'values': [16, 32, 64]
    },
    'weight_initialisation': {
        'values': ['random', 'xavier_normal', 'xavier_uniform']
    },
    'activation_functions': {
        'values': ['sigmoid', 'tanh', 'relu']
    },
}
sweep_config['parameters'] = parameters_dict


def get_name(params):
    keys = [key for key in params.keys()]
    values = [params[key] for key in keys]

    name = ''
    for key, val in zip(keys, values):
        name += ''.join([i[0] for i in key.split('_')])+':'+str(val)+'_'
    return name


def train(config=None):
    import tensorflow as tf
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train = (x_train/255.0).reshape(-1, 784)
    y_train = one_hot(y_train, num_classes=10)

    x_test = (x_test/255.0).reshape(-1, 784)
    y_test = one_hot(y_test, num_classes=10)

    with wandb.init(config=config) as run:
        config = wandb.config
        run.name = get_name(config)

        nn = NeuralNetwork(num_hidden=config.number_of_hidden_layers, num_nodes=config.size_of_every_hidden_layer, weight_decay=config.weight_decay,
                           learning_rate=config.learning_rate, optimizer=config.optimizer, batch_size=config.batch_size, weights_init=config.weight_initialisation, activation=config.activation_functions)
        nn.train((x_train, y_train), (x_test, y_test),
                 epochs=config.number_of_epochs)


if __name__ == '__main__':
    nn = NeuralNetwork()
    import tensorflow as tf
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train = (x_train/255.0).reshape(-1, 784)
    y_train = one_hot(y_train, num_classes=10)
    x_test = (x_test/255.0).reshape(-1, 784)
    y_test = one_hot(y_test, num_classes=10)
    nn.train((x_train, y_train), (x_test, y_test), log=False)

    # wandb.login()
    # sweep_id = wandb.sweep(sweep_config, project='DL_assignment_1')
    # wandb.agent(sweep_id, train, count=250)
    # wandb.finish()
