#include <stdio.h>
#include <cuda.h>

#define N 10



__global__ void array_fill(int *array)
{
    array[threadIdx.x] = threadIdx.x * threadIdx.x;
}

void print_array(int *array, int len)
{
    for (int i = 0; i < len; i++)
    {
        printf("%d \t", array[i]);
    }
    printf("\n");
}

int main()
{
    int a[N] = {0}, *d_a;
    print_array(a, N);

    cudaMalloc(&d_a, sizeof(int) * N);
    array_fill<<<1, N>>>(d_a);
    cudaMemcpy(a, d_a, sizeof(int) * N, cudaMemcpyDeviceToHost);
    cudaFree(d_a);
    print_array(a, N); // cuda device synchronise not needed

    // cudaDeviceSynchronize();
    return 0;
}