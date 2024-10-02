
import numpy as np
import pandas as pd
import tqdm
import os
import matplotlib.pyplot as plt

from dataset import *
from algorithms import *


def solution1(plot=False):
    obj = generate_data('FMLA1Q1Data_train.csv', 'FMLA1Q1Data_test.csv')
    train_data, train_labels, test_data, test_labels = obj.get_data()
    weights, bias = least_square_analytical(train_data, train_labels)

    if plot:
        X = np.linspace(train_data[:, 0].min(), train_data[:, 0].max(), 10)
        Y = np.linspace(train_data[:, 1].min(), train_data[:, 1].max(), 10)
        xx, yy = np.meshgrid(X, Y)
        zz = bias + weights[0]*xx + weights[1]*yy

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.scatter(train_data[:, 0], train_data[:,
                   1], train_labels, c='r', marker='o')
        ax.scatter(test_data[:, 0], test_data[:,
                   1], test_labels, c='b', marker='^')
        ax.plot_surface(xx, yy, zz, alpha=0.5, cmap='plasma',
                        label='Regression plane')
        ax.set_xlabel('X - feature 1')
        ax.set_ylabel('Y - feature 2')
        ax.set_zlabel('Z - label')
        ax.legend()
        plt.tight_layout()
        plt.savefig('images/solution1_3d_plot.png')
        plt.show()

    print('Train error:', mse(train_data, train_labels, weights, bias))
    print('Test error:', mse(test_data, test_labels, weights, bias))

    return weights, bias


def solution2(plot=False):
    obj = generate_data('FMLA1Q1Data_train.csv', 'FMLA1Q1Data_test.csv')
    train_data, train_labels, test_data, test_labels = obj.get_data()
    weights, bias, losses, (weights_history, bias_history) = least_square_gradient_descent(
        train_data, train_labels, epochs=int(20000), learning_rate=1e-4)

    if plot:
        X = np.linspace(train_data[:, 0].min(), train_data[:, 0].max(), 10)
        Y = np.linspace(train_data[:, 1].min(), train_data[:, 1].max(), 10)
        xx, yy = np.meshgrid(X, Y)
        zz = bias + weights[0]*xx + weights[1]*yy

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.scatter(train_data[:, 0], train_data[:,
                   1], train_labels, c='r', marker='o')
        ax.scatter(test_data[:, 0], test_data[:,
                   1], test_labels, c='b', marker='^')
        ax.plot_surface(xx, yy, zz, alpha=0.5, cmap='plasma',
                        label='Regression plane')
        ax.set_xlabel('X - feature 1')
        ax.set_ylabel('Y - feature 2')
        ax.set_zlabel('Z - label')
        ax.legend()
        plt.tight_layout()
        plt.savefig('images/solution2_3d_plot.png')
        plt.show()

        steps = np.arange(0, losses.shape[0])
        plt.plot(steps, losses, label='Loss Linear Regression Gradient Descent')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig('images/solution2_loss_plot.png')
        plt.show()

    print('Train error:', mse(train_data, train_labels, weights, bias))
    print('Test error:', mse(test_data, test_labels, weights, bias))
    return weights, bias, (weights_history, bias_history)


def solution3(plot=False):
    obj = generate_data('FMLA1Q1Data_train.csv', 'FMLA1Q1Data_test.csv')
    train_data, train_labels, test_data, test_labels = obj.get_data()
    weights, bias, losses, (weights_history, bias_history) = least_square_stochastic_gradient_descent(
        train_data, train_labels, epochs=int(20000), learning_rate=1e-4, batch_size=100)

    if plot:
        X = np.linspace(train_data[:, 0].min(), train_data[:, 0].max(), 10)
        Y = np.linspace(train_data[:, 1].min(), train_data[:, 1].max(), 10)
        xx, yy = np.meshgrid(X, Y)
        zz = bias + weights[0]*xx + weights[1]*yy

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.scatter(train_data[:, 0], train_data[:,
                   1], train_labels, c='r', marker='o')
        ax.scatter(test_data[:, 0], test_data[:,
                   1], test_labels, c='b', marker='^')
        ax.plot_surface(xx, yy, zz, alpha=0.5, cmap='plasma',
                        label='Regression plane')
        ax.set_xlabel('X - feature 1')
        ax.set_ylabel('Y - feature 2')
        ax.set_zlabel('Z - label')
        ax.legend()
        plt.tight_layout()
        plt.savefig('images/solution3_3d_plot.png')
        plt.show()

        steps = np.arange(0, losses.shape[0])
        plt.plot(steps, losses,
                 label='Loss Linear Regression Stochastic Gradient Descent')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig('images/solution3_loss_plot.png')
        plt.show()

    print('Train error:', mse(train_data, train_labels, weights, bias))
    print('Test error:', mse(test_data, test_labels, weights, bias))
    return weights, bias, (weights_history, bias_history)


def solution4(plot=False):
    obj = generate_data('FMLA1Q1Data_train.csv', 'FMLA1Q1Data_test.csv')
    train_data, train_labels, test_data, test_labels = obj.get_data()
    lambda_hyperparameters = [1e-4, 1e-3,
                              1e-2, 1e-1, 1e0, 1e1, 1e2]
    validation_loss_lambdas = np.zeros_like(lambda_hyperparameters)
    number_of_folds = 5

    best_lambda = np.inf
    best_error = np.inf

    for ind,lambda_ in tqdm.tqdm(enumerate(lambda_hyperparameters)):
        average_total_train_error = 0
        average_total_validation_error = 0
        for train_data_new, train_labels_new, validation_data, validation_labels in k_fold_split(train_data, train_labels, number_of_folds):
            weights, bias, _ = least_square_ridge_regularization(
                train_data_new, train_labels_new, epochs=int(1e+5), learning_rate=1e-4, lambda_=lambda_)
            average_total_train_error += mse(
                train_data_new, train_labels_new, weights, bias)
            average_total_validation_error += mse(
                validation_data, validation_labels, weights, bias)

        average_total_train_error /= number_of_folds
        average_total_validation_error /= number_of_folds
        validation_loss_lambdas[ind] = average_total_validation_error

        if average_total_validation_error < best_error:
            best_lambda = lambda_
            best_error = average_total_validation_error

    print('Train error:', mse(train_data, train_labels, weights, bias))
    print('Test error:', mse(test_data, test_labels, weights, bias))
    print('Best lambda:', best_lambda)

    if plot:
        X = np.linspace(train_data[:, 0].min(), train_data[:, 0].max(), 10)
        Y = np.linspace(train_data[:, 1].min(), train_data[:, 1].max(), 10)
        xx, yy = np.meshgrid(X, Y)
        zz = bias + weights[0]*xx + weights[1]*yy

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.scatter(train_data[:, 0], train_data[:,
                   1], train_labels, c='r', marker='o')
        ax.scatter(test_data[:, 0], test_data[:,
                   1], test_labels, c='b', marker='^')
        ax.plot_surface(xx, yy, zz, alpha=0.5, cmap='plasma',
                        label='Regression plane')
        ax.set_xlabel('X - feature 1')
        ax.set_ylabel('Y - feature 2')
        ax.set_zlabel('Z - label')
        ax.legend()
        plt.tight_layout()
        plt.savefig('images/solution4_3d_plot.png')
        plt.show()
        
        plt.plot(np.log10(lambda_hyperparameters), validation_loss_lambdas)
        plt.xlabel('Lambda log10')
        plt.ylabel('Validation Loss')
        plt.tight_layout()
        plt.savefig('images/solution4_lambda_plot.png')
        plt.show()
    return weights, bias


def solution5(plot=False):
    obj = generate_data('FMLA1Q1Data_train.csv', 'FMLA1Q1Data_test.csv')
    train_data, train_labels, test_data, test_labels = obj.get_data()
    model = KernelLinearRegresseion(degree=2, coef0=1)
    model.fit(train_data, train_labels)

    if plot:
        X = np.linspace(train_data[:, 0].min(), train_data[:, 0].max(), 10)
        Y = np.linspace(train_data[:, 1].min(), train_data[:, 1].max(), 10)
        xx, yy = np.meshgrid(X, Y)
        zz = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.scatter(train_data[:, 0], train_data[:,
                   1], train_labels, c='r', marker='o')
        ax.scatter(test_data[:, 0], test_data[:,
                   1], test_labels, c='b', marker='^')
        ax.plot_surface(xx, yy, zz, alpha=0.5, cmap='plasma',
                        label='Regression plane')
        ax.set_xlabel('X - feature 1')
        ax.set_ylabel('Y - feature 2')
        ax.set_zlabel('Z - label')
        ax.legend()
        plt.savefig('images/solution5_3d_plot.png')
        plt.show()

    train_predictions = model.predict(train_data)
    test_predictions = model.predict(test_data)
    print('Train error:', np.square(train_labels - train_predictions).mean())
    print('Test error:', np.square(test_labels - test_predictions).mean())
    return model


def plot_weight_history(star, history, name=''):
    star_weight, star_bias = star
    weights_history, bias_history = history
    norm_weights = [l2_norm(star_weight, weights) for weights in weights_history]
    norm_bias = [l2_norm(star_bias, bias) for bias in bias_history]


    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].plot(norm_weights)
    ax[0].set_title('Weight')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('L2 Norm')

    ax[1].plot(norm_bias)
    ax[1].set_title('Bias')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('L2 Norm')

    plt.tight_layout()
    plt.savefig(f'images/{name}.png')
    plt.show()


if __name__ == '__main__':
    os.makedirs('images', exist_ok=True)
    weights_star, bias_star = solution1(plot=True)
    weight, bias, (weights_history, bias_history) = solution2(plot=True)
    plot_weight_history((weights_star, bias_star),
                        (weights_history, bias_history),name='solution2_weight_history')
    print(weights_star,weight, bias_star, bias)
    
    weight, bias, (weights_history, bias_history) = solution3(plot=True)
    plot_weight_history((weights_star, bias_star),
                        (weights_history, bias_history),name='solution3_weight_history')
    print(weights_star,weight, bias_star, bias)
    print(solution4(plot=True))
    solution5(plot=True)
