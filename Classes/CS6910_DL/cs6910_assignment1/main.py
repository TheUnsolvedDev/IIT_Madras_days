import numpy as np
import wandb
from keras.datasets import fashion_mnist

(train_images, train_labels), (_, _) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 
               'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

wandb.init(project='CS23E001_DL_1')

wandb.log({"examples": [wandb.Image(image, caption=class_names[label]) for image, label in zip(train_images[:10], train_labels[:10])]})

