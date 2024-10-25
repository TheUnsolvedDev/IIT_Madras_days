import glob
import os
import torch
import numpy
import torchvision
import pandas as pd
import csv


def make_dataset_csv():
    list_of_files = []
    classes = [i.split('/')[-1] for i in glob.glob('inaturalist_12K/train/*')]
    class_to_index = {clas: i for i, clas in enumerate(classes)}
    index_to_class = {i: clas for i, clas in enumerate(classes)}

    train_files = {i: [] for i in index_to_class.keys()}
    train_list = []
    val_list = []
    test_files = {i: [] for i in index_to_class.keys()}
    test_list = []

    for file in glob.glob('inaturalist_12K/train/*/*'):
        index = class_to_index[file.split('/')[2]]
        train_files[index].append(file)

    for file in glob.glob('inaturalist_12K/val/*/*'):
        index = class_to_index[file.split('/')[2]]
        test_files[index].append(file)

    for clas in index_to_class.keys():
        for file in train_files[clas][:800]:
            train_list.append([file, clas])
        for file in train_files[clas][800:]:
            val_list.append([file, clas])

    for clas in index_to_class.keys():
        for file in test_files[clas][:800]:
            test_list.append([file, clas])

    with open('train.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for row in train_list:
            writer.writerow(row)

    with open('validation.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for row in val_list:
            writer.writerow(row)

    with open('test.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for row in test_list:
            writer.writerow(row)


if __name__ == '__main__':
    make_dataset_csv()
