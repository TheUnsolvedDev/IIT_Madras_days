#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include "cublas_v2.h"

#define n 1000

int main(void)
{
    cudaError_t cuda_stat;
    cublasStatus_t stat;
    cublasHandle_t handle;

    float *x = (float *)malloc(n * sizeof(*x));
    for (int j = 0; j < n; j++)
    {
        x[j] = (float)j;
    }

    printf("x: ");
    for (int j = 0; j < n; j++)
    {
        printf("%2.0f,\t", x[j]);
    }
    printf("\n");

    float *d_x;
    cuda_stat = cudaMalloc((void **)&d_x, n * sizeof(*d_x));
    stat = cublasCreate(&handle);
    stat = cublasSetVector(n, sizeof(*x), x, 1, d_x, 1);
    
    float result;
    stat = cublasSasum(handle, n, d_x, 1, &result);
    printf("sum |x[i]|: \t %4.0f\n", result);

    cudaFree(d_x);
    cublasDestroy(handle);
    free(x);

    return 0;
}
