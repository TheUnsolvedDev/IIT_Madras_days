#include <stdio.h>
#include <math.h>
#include <limits.h>

extern "C"
{
#include "initializers.h"
#include "activations.cuh"
}

void activation_present()
{
    printf("Activation Present\n");
}

__device__ float sigmoid(float x)
{
    return 1.0 / (1.0 + exp(-x));
}

__global__ void sigmoid_kernel(float *input, float *output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size)
    {
        output[idx] = sigmoid(input[idx]);
    }
}

random_weights sigmoid_activation(random_weights rw)
{
    int total_data = rw.size[0] * rw.size[1];
    unsigned int num_threads = 256;
    unsigned int num_blocks = ceil((float)total_data / num_threads);

    float *dvector, *dres_vector, *vector = convert2DTo1D(rw.weight, rw.size[0], rw.size[1]);
    cudaMalloc(&dvector, total_data * sizeof(float));
    cudaMalloc(&dres_vector, total_data * sizeof(float));

    cudaMemcpy(dvector, vector, total_data * sizeof(float), cudaMemcpyHostToDevice);
    sigmoid_kernel<<<num_blocks, num_threads>>>(dvector, dres_vector, total_data);
    cudaMemcpy(vector, dres_vector, total_data * sizeof(float), cudaMemcpyDeviceToHost);

    rw.weight = convert1DTo2D(vector, rw.size[0], rw.size[1]);

    cudaFree(dvector);
    cudaFree(dres_vector);
    return rw;
}