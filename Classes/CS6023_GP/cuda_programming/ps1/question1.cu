#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define MAX_THREADS 512

__global__ void add_gpu(float *vector1, float *vector2, float *result, int length)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < length)
    {
        result[idx] = vector1[idx] + vector2[idx];
        // printf("%f %f %f\n", result[idx], vector1[idx], vector2[idx]);
    }
}

void print_array(float *vector, int n)
{
    for (int i = 0; i < n; i++)
    {
        printf("%.4f\t", vector[i]);
    }
    printf("\n");
}

int main()
{
    int num_elements = 20000;
    int num_blocks = ceil((float)num_elements / (float)MAX_THREADS);
    printf("NUM BLOCKS:%d\n", num_blocks);

    float vector1[num_elements] = {0};
    float vector2[num_elements] = {0};

    for (int i = 0; i < num_elements; i++)
    {
        vector1[i] = (float)(i % 100);
        vector2[i] = (float)(i % 50);
    }

    print_array(vector1, num_elements);
    print_array(vector2, num_elements);

    float *dvector1, *dvector2, *dresult;
    float *result = (float *)malloc(num_elements * sizeof(float));

    cudaMalloc(&dvector1, num_elements * sizeof(float));
    cudaMalloc(&dvector2, num_elements * sizeof(float));
    cudaMalloc(&dresult, num_elements * sizeof(float));

    cudaMemcpy(dvector1, vector1, num_elements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dvector2, vector2, num_elements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dresult, dresult, num_elements * sizeof(float), cudaMemcpyHostToDevice);

    add_gpu<<<num_blocks, MAX_THREADS>>>(dvector1, dvector2, dresult, num_elements);
    // cudaDeviceSynchronize();

    cudaMemcpy(result, dresult, num_elements * sizeof(float), cudaMemcpyDeviceToHost);
    print_array(result, num_elements);

    free(result);
    cudaFree(dvector1);
    cudaFree(dvector2);
    cudaFree(dresult);
    return 0;
}