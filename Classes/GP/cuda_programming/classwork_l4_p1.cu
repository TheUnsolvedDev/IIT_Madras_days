#include <stdio.h>
#include <cuda.h>

__global__ void init(int *array, int a_length)
{
    unsigned int id = threadIdx.x;
    if (id < a_length)
    {
        array[id] = 0;
    }
}

__global__ void add(int *array, int a_length)
{
    unsigned id = threadIdx.x;
    if (id < a_length)
    {
        array[id] += id;
    }
}

void print_list(int *array, int a_length)
{
    for (int i = 0; i < a_length; i++)
        printf("%d \t", array[i]);
    printf("\n");
}


int main()
{
    int *d_a;
    int n = 1024;
    cudaMalloc(&d_a, n * sizeof(int));

    init<<<1, n>>>(d_a, n);
    add<<<1, n>>>(d_a, n);
    cudaDeviceSynchronize();

    int array[n];
    cudaMemcpy(array, d_a, n * sizeof(int), cudaMemcpyDeviceToHost);

    print_list(array, n);
    cudaFree(d_a);

    return 0;
}