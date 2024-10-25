# cs6910_assignment2

## **Deep Learning Assignment 2**

Inside this repository lie two parts:

1. Part A focusses on training a CNN from scratch.
2. Part B focusses on fine-tuning parameters from a pre-existing model.

The assignment is implemented using higher level APIs like Keras. The following links describe the problem statement and the results:

1. [Problem Statement](https://wandb.ai/cs6910_2024_mk/A1/reports/CS6910-Assignment-2--Vmlldzo3MjcwNzM1)
2. [Report](https://wandb.ai/shuvrajeet/CS23E001_DL_2/reports/CS6910-Assignment-2--Vmlldzo3NDA0NjIw)

## Contents

## Part A

The above file contains the implementation of training a CNN in Keras. The sweep features configured for this are:

1. Shape of the filters ('kernel_size') : [(5,5) and (3,3)]
2. L2 regularisation ('weight_decay') : [0, 0.0005, 0.005]
3. Dropout ('dropout') : [0, 0.2, 0.4]
4. Learning rate ('learning_rate') : [1e-3, 1e-4]
5. Activation function ('activation') : ['relu', 'elu', 'selu']
6. Batch size for training ('batch_size') : 64,
7. Batch Normalisation ('batch_norm') : ['true','false']
8. Filter organisation ('filt_org' ) : ['same','half','double']
9. Data augmentation ('data_augment') : ['true','false']
10. Number of neurons in dense layer ('num_dense') : [32, 64, 128, 256]

## Part B

The methods used are:

1. Training from scratch
2. Training by  fine tuning only hidden layers
3. Training by initialising with Image net weights


## Sweep configurations

1. Method: Bayes
2. Metric: Validation accuracy (to be maximised)
3. Parameters (mentioned above)

## Running the script
```bash

# Running for part A question answers

cd PartA
python3 train_parta.py --sweep # runs a sweep containing 100 runs
python3 train_parta.py -nf 64 -act Mish -fo half -da True -bn True -dr 0.45 # describing how to run for a single hyperparameter setup
python3 train_parta.py --help # for help

# Running for part B question answers

cd PartB
python3 train_partb.py -ft True -pt True # describing how to run for single hyperparameter setup
python3 train_partb.py --help # for help

```