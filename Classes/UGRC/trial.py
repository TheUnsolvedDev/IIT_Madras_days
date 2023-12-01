import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import numpy as np

# Check if multiple GPUs are available
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Enable GPU memory growth to allocate only as needed
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Define a MirroredStrategy to use multiple GPUs
strategy = tf.distribute.MirroredStrategy()

# Define your model within the strategy scope
with strategy.scope():
    # Define your model here (for demonstration, a simple feedforward network)
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer=optimizers.Adam(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

# Create some example data (MNIST dataset for demonstration)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Distribute the dataset across GPUs (using tf.data.Dataset)
train_dataset = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).batch(4096)

# Train the model using the distributed dataset
with strategy.scope():
    model.fit(train_dataset, epochs=50)

# Evaluate the model (you can do this outside the strategy scope)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(64)
eval_result = model.evaluate(test_dataset)
print("Test loss:", eval_result[0])
print("Test accuracy:", eval_result[1])
