#!/bin/bash

# Define arrays of possible values for each parameter
reconstructions=(
    # 'without_regularization_and_zero_prior'
    # 'without_regularization_and_sparsity_prior'
    # 'with_tikhonov_and_identity_and_zero_prior'
    # 'with_tikhonov_and_identity_and_sparsity_prior'
    # 'with_tikhonov_and_lambda_zero_prior'
    # 'with_tikhonov_and_lambda_sparsity_prior'
    # 'with_Sharpen_regularization_and_zero_prior'
    # 'with_Sharpen_regularization_and_sparsity_prior'
    # 'with_Sharpen_regularization_and_lambda_zero_prior'
    # 'with_Sharpen_regularization_and_lambda_sparsity_prior'
    # 'with_LOG_regularization_and_zero_prior'
    # 'with_LOG_regularization_and_sparsity_prior'
    # 'with_LOG_regularization_and_lambda_zero_prior'
    # 'with_LOG_regularization_and_lambda_sparsity_prior'
    # 'with_Gauss_regularization_and_zero_prior'
    # 'with_Gauss_regularization_and_sparsity_prior'
    # 'with_Gauss_regularization_and_lambda_zero_prior'
    # 'with_Gauss_regularization_and_lambda_sparsity_prior'
    'with_Laplacian_regularization'
)
strategies=("all")
sizes=(10 16 25)
num_maps=(5)

# Iterate through each combination of parameters
for recon in "${reconstructions[@]}"; do
    for strat in "${strategies[@]}"; do
        for size in "${sizes[@]}"; do
            for num_map in "${num_maps[@]}"; do
                echo "Running: main.py -r $recon -st $strat -s $size -nm $num_map"
                python main.py -r "$recon" -st "$strat" -s "$size" -nm "$num_map"
            done
        done
    done
done
