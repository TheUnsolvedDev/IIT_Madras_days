// kernel.cu
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

extern "C"
{
#include "initializers.h"
#include "matmul.cuh"
#include "utils.h"
}

__global__ void matmul_present_kernel()
{
    printf("Matmul present\n");
}

extern "C" void matmul_present()
{
    matmul_present_kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
}

__global__ void matrix_multiply_kernel(float *a, float *b, float *result, int m, int n, int k)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < k)
    {
        float sum = 0.0f;
        for (int i = 0; i < n; ++i)
        {
            sum += a[row * n + i] * b[i * k + col];
        }
        result[row * k + col] = sum;
    }
}

tensor matrix_multiply(tensor a, tensor b)
{
    int m = a.size[0];
    int n = a.size[1];
    int k = b.size[0];

    if (a.size[1] != b.size[0])
    {
        printf("Illegal dimension %d != %d", a.size[1], b.size[0]);
        exit(EXIT_FAILURE);
    }

    tensor result = allocate_zero_weights(m, k);
    float *da, *db, *dresult;
    float *vector_a = convert2DTo1D(a.matrix, a.size[0], a.size[1]);
    float *vector_b = convert2DTo1D(b.matrix, b.size[0], b.size[1]);
    float *res = (float *)malloc(m * k * sizeof(float));

    cudaMalloc(&da, m * n * sizeof(float));
    cudaMalloc(&db, n * k * sizeof(float));
    cudaMalloc(&dresult, m * k * sizeof(float));

    // cudaMemcpy(da, vector_a, m * n * sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(db, vector_b, n * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(da, &(a.matrix[0][0]), m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(db, &(b.matrix[0][0]), n * k * sizeof(float), cudaMemcpyHostToDevice);

    dim3 dim_grid(ceilf(m / (float)NUM_2D_THREADS), ceilf(k / (float)NUM_2D_THREADS));
    dim3 dim_block(NUM_2D_THREADS, NUM_2D_THREADS);

    matrix_multiply_kernel<<<dim_grid, dim_block>>>(da, db, dresult, m, n, k);
    cudaMemcpy(&(result.matrix[0][0]), dresult, m * k * sizeof(float), cudaMemcpyDeviceToHost);

    // result.matrix = convert1DTo2D(res, m, k);

    free(vector_a);
    free(vector_b);
    cudaFree(da);
    cudaFree(db);
    cudaFree(dresult);
    return result;
}