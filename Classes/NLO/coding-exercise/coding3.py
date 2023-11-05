import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time


class Solutions:
    x_star = np.array([1, 1])
    num_data_points = 100

    def __init__(self):
        print('''
              The solutions are presented as follows:
              ''')

    @staticmethod
    def linear_model(x, w, b):
        return x*w + b

    @staticmethod
    def error(y_true, y_pred):
        squared_error = np.square(y_true - y_pred)
        return squared_error.mean()

    @staticmethod
    def grads(x, y, w, b):
        w_grad = -2*(y - Solutions.linear_model(x, w, b))*x
        b_grad = -2*(y - Solutions.linear_model(x, w, b))
        return w_grad.mean(), b_grad.mean()

    @staticmethod
    def linear_regression_gradient_descent(data, labels, steps, alpha, history=None):
        x_random = np.array([np.random.normal() for _ in range(len(Solutions.x_star))])

        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(data, labels)

        line1, = ax.plot(data, Solutions.linear_model(data, *x_random), color='red')
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.title("Linear Regression with Gradient Descent")

        for _ in range(steps):
            pred = Solutions.linear_model(data, *x_random)
            loss = Solutions.error(labels, pred)

            if history:
                history['gd']['w'].append(x_random[0])
                history['gd']['b'].append(x_random[1])

            print('[{}/{}]\t Current Loss is {:.3f}\t'.format(_, steps, loss))
            w_grad, b_grad = Solutions.grads(data, labels, *x_random)
            x_random[0] -= alpha*w_grad
            x_random[1] -= alpha*b_grad

            line1.set_xdata(data)
            line1.set_ydata(pred)
            fig.canvas.draw()
            fig.canvas.flush_events()
            # time.sleep(0.01)

        plt.title("Linear Regression with Gradient Descent. Done Training!!")
        plt.show(block=True)

        return x_random

    @staticmethod
    def linear_regression_stochastic_gradient_descent(data, labels, steps, alpha, S=10, history=None):
        x_random = np.array([np.random.normal() for _ in range(len(Solutions.x_star))])

        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(data, labels)

        line1, = ax.plot(data, Solutions.linear_model(data, *x_random), color='red')
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.title("Linear Regression with Stochastic Gradient Descent")

        num_batches = len(data) // S
        for _ in range(steps):
            batch_indices = np.arange(num_batches)
            np.random.shuffle(batch_indices)
            for batch_index in batch_indices:
                batched_data = data[(batch_index)*S:(batch_index+1)*S]
                batched_labels = labels[(batch_index)*S:(batch_index+1)*S]
                pred = Solutions.linear_model(batched_data, *x_random)
                loss = Solutions.error(batched_labels, pred)

                if history:
                    history['sgd']['w'].append(x_random[0])
                    history['sgd']['b'].append(x_random[1])

                w_grad, b_grad = Solutions.grads(batched_data, batched_labels, *x_random)
                x_random[0] -= alpha*w_grad
                x_random[1] -= alpha*b_grad
            print('[{}/{}]\t Current Loss is {:.3f}\t'.format(_, steps, loss))

            line1.set_xdata(data)
            line1.set_ydata(Solutions.linear_model(data, *x_random))
            fig.canvas.draw()
            fig.canvas.flush_events()
            # time.sleep(0.01)

        plt.title("Linear Regression with Stochastic Gradient Descent. Done Training!!")
        plt.show(block=True)

        return x_random

    def exercise1(self):
        total_steps = [100, 1000]
        alphas = [0.01, 0.1]

        data = np.linspace(0, 1, 100)
        labels = self.linear_model(data, *self.x_star) + np.random.uniform(-1, 1)

        # self.linear_regression_gradient_descent(data, labels, total_steps[0], alphas[0])
        # self.linear_regression_stochastic_gradient_descent(data, labels, total_steps[0], alphas[0])

        # self.exercise1a(data, labels)
        # self.exercise1b(data, labels)
        # self.exercise1c(data, labels)
        # self.exercise1d(data, labels)
        self.exercise1e(data, labels)

    def exercise1a(self, data, labels):
        # subquestion (a)
        alpha = 0.01
        history = {
            'gd': {'w': [], 'b': []},
            'sgd': {'w': [], 'b': []}
        }
        self.linear_regression_gradient_descent(data, labels, 250, alpha, history=history)
        self.linear_regression_stochastic_gradient_descent(data, labels, 250, alpha, history=history)

        fig, axs = plt.subplots(1, 2, figsize=(16, 8))
        axs[0].plot(np.sqrt(np.square(self.x_star[0] - history['gd']['w'])), label='GD weight')
        axs[0].plot(np.sqrt(np.square(self.x_star[1] - history['gd']['b'])), label='GD bias')
        axs[0].set(ylabel=r'$\|x_t - x_{*}\|_{2}$', xlabel='step')
        axs[0].grid()
        axs[0].legend()

        axs[1].plot(np.sqrt(np.square(self.x_star[0] - history['sgd']['w'])), label='SGD weight')
        axs[1].plot(np.sqrt(np.square(self.x_star[1] - history['sgd']['b'])), label='SGD bias')
        axs[1].set(ylabel=r'$\|x_t - x_{*}\|_{2}$', xlabel='step')
        axs[1].grid()
        axs[1].legend()

        plt.title(r'$\alpha$=0.01')
        plt.show(block=True)

    def exercise1b(self, data, labels):
        # subquestion (b)
        alpha = 0.1
        history = {
            'gd': {'w': [], 'b': []},
            'sgd': {'w': [], 'b': []}
        }
        self.linear_regression_gradient_descent(data, labels, 250, alpha, history=history)
        self.linear_regression_stochastic_gradient_descent(data, labels, 250, alpha, history=history)

        fig, axs = plt.subplots(1, 2, figsize=(16, 8))
        axs[0].plot(np.sqrt(np.square(self.x_star[0] - history['gd']['w'])), label='GD weight')
        axs[0].plot(np.sqrt(np.square(self.x_star[1] - history['gd']['b'])), label='GD bias')
        axs[0].set(ylabel=r'$\|x_t - x_{*}\|_{2}$', xlabel='step')
        axs[0].grid()
        axs[0].legend()

        axs[1].plot(np.sqrt(np.square(self.x_star[0] - history['sgd']['w'])), label='SGD weight')
        axs[1].plot(np.sqrt(np.square(self.x_star[1] - history['sgd']['b'])), label='SGD bias')
        axs[1].set(ylabel=r'$\|x_t - x_{*}\|_{2}$', xlabel='step')
        axs[1].grid()
        axs[1].legend()

        plt.title(r'$\alpha$=0.1')
        plt.show(block=True)

    def exercise1c(self, data, labels):
        # subquestion (c)
        total_steps = 1000
        alpha = 0.1
        history = {
            'gd': {'w': [], 'b': []},
            'sgd': {'w': [], 'b': []}
        }
        self.linear_regression_gradient_descent(data, labels, total_steps, alpha, history=history)
        self.linear_regression_stochastic_gradient_descent(data, labels, total_steps, alpha, history=history)

        fig, axs = plt.subplots(1, 2, figsize=(16, 8))
        axs[0].plot(np.sqrt(np.square(self.x_star[0] - history['gd']['w'])), label='GD weight')
        axs[0].plot(np.sqrt(np.square(self.x_star[1] - history['gd']['b'])), label='GD bias')
        axs[0].set(ylabel=r'$\|x_t - x_{*}\|_{2}$', xlabel='step')
        axs[0].grid()
        axs[0].legend()

        axs[1].plot(np.sqrt(np.square(self.x_star[0] - history['sgd']['w'])), label='SGD weight')
        axs[1].plot(np.sqrt(np.square(self.x_star[1] - history['sgd']['b'])), label='SGD bias')
        axs[1].set(ylabel=r'$\|x_t - x_{*}\|_{2}$', xlabel='step')
        axs[1].grid()
        axs[1].legend()

        plt.title(r'$\alpha$=0.1')
        plt.show(block=True)

    def exercise1d(self, data, labels):
        # subquestion (d)
        total_steps = 1000
        alpha = 0.01
        history = {
            'gd': {'w': [], 'b': []},
            'sgd': {'w': [], 'b': []}
        }
        self.linear_regression_gradient_descent(data, labels, total_steps, alpha, history=history)
        self.linear_regression_stochastic_gradient_descent(data, labels, total_steps, alpha, history=history)

        fig, axs = plt.subplots(1, 2, figsize=(16, 8))
        axs[0].plot(np.sqrt(np.square(self.x_star[0] - history['gd']['w'])), label='GD weight')
        axs[0].plot(np.sqrt(np.square(self.x_star[1] - history['gd']['b'])), label='GD bias')
        axs[0].set(ylabel=r'$\|x_t - x_{*}\|_{2}$', xlabel='step')
        axs[0].grid()
        axs[0].legend()

        axs[1].plot(np.sqrt(np.square(self.x_star[0] - history['sgd']['w'])), label='SGD weight')
        axs[1].plot(np.sqrt(np.square(self.x_star[1] - history['sgd']['b'])), label='SGD bias')
        axs[1].set(ylabel=r'$\|x_t - x_{*}\|_{2}$', xlabel='step')
        axs[1].grid()
        axs[1].legend()

        plt.title(r'$\alpha$=0.01')
        plt.show(block=True)

    def exercise1e(self, data, labels):
        # subquestion (e)
        total_steps = 1000
        alpha = 0.01

        w_s = np.linspace(-2, 2, 100)
        b_s = np.linspace(-2, 2, 100)
        W, B = np.meshgrid(w_s, b_s)

        def error_fn(w, b): return np.square(labels - (w*data + b)).mean()
        errors = np.array([error_fn(w, b) for w, b in zip(W.flatten(), B.flatten())]).reshape(W.shape)

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.plot_surface(W, B, errors, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)
        plt.show(block=True)


if __name__ == '__main__':
    answers = Solutions()
    answers.exercise1()
