#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include "cublas_v2.h"

#define n 1000

void print_array(float *array, int length)
{
    for (int j = 0; j < length; j++)
    {
        printf("%2.0f,\t", array[j]);
    }
    printf("\n");
}

int main(void)
{
    cudaError_t cuda_stat;
    cublasStatus_t stat;
    cublasHandle_t handle;

    float *x = (float *)malloc(n * sizeof(*x));
    for (int j = 0; j < n; j++)
    {
        x[j] = (float)j + (float)(rand() % n);
    }
    float *y = (float *)malloc(n * sizeof(*y));
    for (int j = 0; j < n; j++)
    {
        y[j] = (float)j + (float)(rand() % n);
    }

    printf("x: ");
    print_array(x, n);

    printf("y: ");
    print_array(y, n);

    float *d_x, *d_y;
    cuda_stat = cudaMalloc((void **)&d_x, n * sizeof(*d_x));
    cuda_stat = cudaMalloc((void **)&d_y, n * sizeof(*d_y));

    stat = cublasCreate(&handle);
    stat = cublasSetVector(n, sizeof(*x), x, 1, d_x, 1);
    stat = cublasSetVector(n, sizeof(*y), y, 1, d_y, 1);
    float a = 2.0;

    stat = cublasSaxpy(handle, n, &a, d_x, 1, d_y, 1);
    stat = cublasGetVector(n, sizeof(float), d_y, 1, y, 1);
    printf("y = ax + y: ");
    print_array(y, n);

    cudaFree(d_x);
    cudaFree(d_y);
    cublasDestroy(handle);
    free(x);
    free(y);

    return 0;
}
