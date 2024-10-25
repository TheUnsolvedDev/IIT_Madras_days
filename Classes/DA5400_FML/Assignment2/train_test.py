import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import argparse


from dataset import dataset
from algorithm import NaiveBayes
from algorithm import LogisticRegression
from algorithm import KNearestNeighbors
from algorithm import DecisionTree

if __name__ == '__main__':
    train_data, train_labels, test_data, test_labels = dataset()
    model = NaiveBayes()
    model.fit(train_data, train_labels)
    
    y_pred_train = model.predict(train_data)
    y_pred_test = model.predict(test_data)

    accuracy_train = np.sum(y_pred_train == train_labels) / len(train_labels)
    accuracy_test = np.sum(y_pred_test == test_labels) / len(test_labels)

    print('Accuracy on training set: ', accuracy_train)
    print('Accuracy on test set: ', accuracy_test)
    
    model = LogisticRegression()
    model.fit(train_data, train_labels)
    
    y_pred_train = model.predict(train_data)
    y_pred_test = model.predict(test_data)

    accuracy_train = np.sum(y_pred_train == train_labels) / len(train_labels)
    accuracy_test = np.sum(y_pred_test == test_labels) / len(test_labels)

    print('Accuracy on training set: ', accuracy_train)
    print('Accuracy on test set: ', accuracy_test)
    
    model = KNearestNeighbors()
    model.fit(train_data, train_labels)
    
    y_pred_train = model.predict(train_data)
    y_pred_test = model.predict(test_data)

    accuracy_train = np.sum(y_pred_train == train_labels) / len(train_labels)
    accuracy_test = np.sum(y_pred_test == test_labels) / len(test_labels)

    print('Accuracy on training set: ', accuracy_train)
    print('Accuracy on test set: ', accuracy_test)
    
    model = DecisionTree()
    model.fit(train_data, train_labels)
    
    y_pred_train = model.predict(train_data)
    y_pred_test = model.predict(test_data)

    accuracy_train = np.sum(y_pred_train == train_labels) / len(train_labels)
    accuracy_test = np.sum(y_pred_test == test_labels) / len(test_labels)

    print('Accuracy on training set: ', accuracy_train)
    print('Accuracy on test set: ', accuracy_test)
    
    model.render_tree()