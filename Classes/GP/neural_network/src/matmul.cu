// kernel.cu
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

extern "C"
{
#include "matmul.cuh"
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
