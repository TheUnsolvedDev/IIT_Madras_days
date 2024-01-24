#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define BLOCK_SIZE 1024

__host__ float mean_of_array(float *array, int length)
{
    float mean = 0.0f;
    for (int i = 0; i < length; i++)
    {
        mean += array[i];
    }
    return mean / (float)length;
}

__global__ void mean_of_array_gpu(float *array, int length)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = length / 2;

    while (stride >= 1)
    {
        if (idx < stride)
        {
            array[idx] += array[idx + stride];
        }
        stride /= 2;
        __syncthreads();
    }

    if (idx == 0)
    {
        array[0] = array[0] / (float)length;
    }
}

int main()
{
    int length = pow(2, 15);
    float array[length] = {0};
    for (int i = 0; i < length; i++)
    {
        array[i] = i + 1;
    }
    printf("Mean: %.4f\n", mean_of_array(array, length));

    float result, *darray;
    int nblocks = ceil((float)length / BLOCK_SIZE);

    cudaMalloc(&darray, length * sizeof(float));
    cudaMemcpy(darray, array, length * sizeof(float), cudaMemcpyHostToDevice);

    mean_of_array_gpu<<<nblocks, BLOCK_SIZE>>>(darray, length);
    cudaDeviceSynchronize();

    cudaMemcpy(array, darray, length * sizeof(float), cudaMemcpyDeviceToHost);
    result = array[0];
    printf("Mean GPU: %.4f\n", result);

    cudaFree(darray);

    return 0;
}