import numpy as np
import torch
import wandb
import os
import argparse

# Import custom modules
from config import *
from model import *
from dataset import *

# Set CUDA visible devices
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_name(params):
    """
    Generate a name for the experiment based on the given parameters.

    Parameters:
    - params (dict): Dictionary containing experiment parameters.

    Returns:
    - str: Name for the experiment.
    """
    keys = [key for key in params.keys()]
    values = [params[key] for key in keys]

    name = ''
    for key, val in zip(keys, values):
        name += ''.join([i[0] for i in key.split('_')])+':'+str(val)+'_'
    return name


def model_parser(num_filters_for_each_layer: int, activation_function: str, filter_organisation: str, data_augmentation: bool, batch_normalisation: bool, dropout_rate: float, filter_size: int, dense_number: int):
    """
    Parse and train the model based on the given hyperparameters.

    Parameters:
    - num_filters_for_each_layer (int): Number of filters for each convolutional layer.
    - activation_function (str): Activation function to be used.
    - filter_organisation (str): Organization of filters (same, double, half).
    - data_augmentation (bool): Whether to use data augmentation.
    - batch_normalisation (bool): Whether to use batch normalisation.
    - dropout_rate (float): Dropout rate.
    - filer_size (int): Size of the filters
    - dense_number (int): Size of the dense connections
    """
    data = Dataset(transformation=data_augmentation)
    if activation_function == 'ReLU':
        activation = torch.nn.ReLU()
    elif activation_function == 'GeLU':
        activation = torch.nn.GELU()
    elif activation_function == 'SiLU':
        activation = torch.nn.SiLU()
    elif activation_function == 'Mish':
        activation = torch.nn.Mish()
    else:
        print('Enter correct activation')
        exit(0)

    if filter_organisation == 'same':
        filters = [num_filters_for_each_layer for i in range(5)]
    elif filter_organisation == 'half':
        filters = [int(num_filters_for_each_layer/(2**(i))) for i in range(5)]
    elif filter_organisation == 'double':
        filters = [int(num_filters_for_each_layer*(2**(i))) for i in range(5)]
    else:
        print('Enter correct filter organisation')
        exit(0)

    if filter_size >= 11:
        conv_filters_sizes = [11, 7, 5, 3, 3]
        conv_filter_strides = [2, 2, 2, 1, 1]
    if filter_size >= 7:
        conv_filters_sizes = [7, 7, 5, 3, 3]
        conv_filter_strides = [1, 1, 1, 1, 1]
    else:
        conv_filters_sizes = [5, 5, 5, 3, 3]
        conv_filter_strides = [1, 1, 1, 1, 1]

    dataset = Dataset(transformation=data_augmentation)
    model = CNN(conv_filters=filters, dropout_rate=dropout_rate, activation=activation, batch_normalization=batch_normalisation,
                conv_kernel_sizes=conv_filters_sizes, conv_strides=conv_filter_strides, dense_units=dense_number).to(device)
    train, val, test = dataset.get_data()
    training(model, [train, val])


def train_sweep(config=None):
    """
    Perform hyperparameter sweep using WandB.

    Parameters:
    - config (dict): Configuration dictionary for hyperparameter sweep.
    """
    with wandb.init(config=config) as run:
        config = wandb.config
        run.name = get_name(config)
        model_parser(
            num_filters_for_each_layer=config.num_filters,
            activation_function=config.activation,
            filter_organisation=config.filter_organisation,
            data_augmentation=config.data_augmentation,
            batch_normalisation=config.batch_normalisation,
            dropout_rate=config.dropout_rate,
            filter_size=7,
            dense_number=256
        )


def train(args):
    """
    Train the model with the specified hyperparameters.

    Parameters:
    - args (argparse.Namespace): Parsed command-line arguments.
    """
    print(args)
    with wandb.init(project=args.wandb_project, entity=args.wandb_entity) as run:
        config = wandb.config
        config.num_filters = args.num_filters
        config.activation = args.activation
        config.filter_organisation = args.filter_organisation
        config.data_augmentation = args.data_augmentation
        config.batch_normalisation = args.batch_normalisation
        config.dropout_rate = args.dropout_rate
        run.name = get_name(config)
        model_parser(
            num_filters_for_each_layer=config.num_filters,
            activation_function=config.activation,
            filter_organisation=config.filter_organisation,
            data_augmentation=config.data_augmentation,
            batch_normalisation=config.batch_normalisation,
            dropout_rate=config.dropout_rate,
            filter_size=args.filter_size,
            dense_number=args.dense,
        )


def main():
    """
    Main function for training neural network with specified hyperparameters.
    """
    parser = argparse.ArgumentParser(
        description="Train neural network with specified hyperparameters")
    parser.add_argument("-wp", "--wandb_project", default="CS23E001_DL_2",
                        help="Project name used to track experiments in Weights & Biases dashboard")
    parser.add_argument("-we", "--wandb_entity", default="Shuvrajeet",
                        help="defines the enitity of the project")
    parser.add_argument("--sweep", action="store_true",
                        help="Perform hyperparameter sweep using wandb")
    parser.add_argument('-nf', '--num_filters', default=32, type=int, choices=[32, 64, 128, 256],
                        help='Number of filters for each layer')
    parser.add_argument('-act', '--activation', default='ReLU', type=str,
                        choices=['ReLU', 'GeLU', 'SiLU', 'Mish'], help='Choice of activation function')
    parser.add_argument('-fo', '--filter_organisation', type=str,
                        default='double', choices=['same', 'double', 'half'], help='Choice of filter organisation')
    parser.add_argument('-da', '--data_augmentation', type=bool, default=True,
                        help='Use of data augmentation')
    parser.add_argument('-bn', '--batch_normalisation', type=bool, default=True,
                        help='Use of batch normalisation')
    parser.add_argument('-dr', '--dropout_rate', type=float, default=0.3,
                        help='Dropout rate value')
    parser.add_argument('-fs', '--filter_size', default=7,
                        type=int, help='Size of the filter')
    parser.add_argument('-d', '--dense', default=256,
                        type=int, help='Size of dense layer')
    args = parser.parse_args()

    if args.sweep:
        sweep_config = {
            'method': 'bayes',
        }

        metric = {
            'name': 'val_accuracy',
            'goal': 'maximize'
        }
        sweep_config['metric'] = metric
        parameters_dict = {
            'num_filters': {
                'values': [32, 64, 128]
            },
            'activation': {
                'values': ['ReLU', 'GeLU', 'SiLU', 'Mish']
            },
            'filter_organisation': {
                'values': ['same', 'double', 'half']
            },
            'data_augmentation': {
                'values': [True, False]
            },
            'batch_normalisation': {
                'values': [True, False]
            },
            'dropout_rate': {
                'values': [i/10 for i in range(2, 5)]
            }
        }
        sweep_config['parameters'] = parameters_dict

        sweep_id = wandb.sweep(
            sweep_config, project=args.wandb_project)
        wandb.agent(sweep_id, function=train_sweep, count=100)
    else:
        train(args)


if __name__ == "__main__":
    main()
