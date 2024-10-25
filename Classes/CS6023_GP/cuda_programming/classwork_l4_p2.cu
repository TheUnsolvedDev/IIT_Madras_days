#include <stdio.h>
#include <cuda.h>

void print_list(int *array, int a_length)
{
    for (int i = 0; i < a_length; i++)
        printf("%d \t", array[i]);
    printf("\n");
}

__global__ void multi_dim_kernel()
{
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) // && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0)
    {
        printf("The value is %d %d %d, %d %d %d\n", gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z);
    }
}

int main()
{
    dim3 grid(2, 3, 4);
    dim3 block(5, 6, 7);
    multi_dim_kernel<<<grid, block>>>();
    cudaDeviceSynchronize();

    return 0;
}