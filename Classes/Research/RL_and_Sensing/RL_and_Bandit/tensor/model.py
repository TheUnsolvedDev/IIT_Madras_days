import tensorflow as tf
import numpy as np

from config import *

def QModel(inputs = (ENV_SIZE, ENV_SIZE, 2),outputs = 10):
    inputs = tf.keras.layers.Input(shape=inputs)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    # x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
    # x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    outputs = tf.keras.layers.Dense(outputs)(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)

if __name__ == '__main__':
    model = QModel()
    model.summary()
    
    print(model(np.ones((1,ENV_SIZE,ENV_SIZE,2))))