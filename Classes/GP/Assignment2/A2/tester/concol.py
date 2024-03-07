import os
import numpy as np
import tensorflow as tf


def read_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        m, n, k = map(int, lines[0].split())
        image_data = np.array([[int(val) for val in line.split()]
                              for line in lines[1:m+1]])
        filter_data = np.array(
            [[int(val) for val in line.split()] for line in lines[m+1:]])
    return m, n, k, image_data, filter_data


def perform_convolution(image_data, filter_data):
    image_data = tf.constant(image_data, dtype=tf.float32)
    filter_data = tf.constant(filter_data, dtype=tf.float32)
    image_data = tf.expand_dims(tf.expand_dims(image_data, 0), -1)
    filter_data = tf.reshape(
        filter_data, [filter_data.shape[0], filter_data.shape[1], 1, 1])
    convolved_image = tf.nn.conv2d(image_data, filter_data, strides=[
                                   1, 1, 1, 1], padding='SAME')
    return convolved_image.numpy().squeeze().astype(np.int32)


def write_to_file(file_path, data):
    with open(file_path.replace('test','out'), 'w') as file:
        for row in data:
            file.write(' '.join(map(str, row)) + '\n')


def convolution_on_files(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for file_name in sorted(os.listdir(input_folder))[6:]:
        if file_name.endswith('.txt'):
            print(file_name)
            input_file_path = os.path.join(input_folder, file_name)
            output_file_path = os.path.join(output_folder, file_name)
            m, n, k, image_data, filter_data = read_file(input_file_path)
            result = perform_convolution(image_data, filter_data)
            write_to_file(output_file_path, result)


# Example usage:
input_folder = 'input'
output_folder = 'result'
convolution_on_files(input_folder, output_folder)
