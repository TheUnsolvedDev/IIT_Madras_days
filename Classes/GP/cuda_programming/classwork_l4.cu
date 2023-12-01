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
    int N = 1024;
    cudaMalloc(&d_a, N * sizeof(int));

    init<<<1, N>>>(d_a, N);
    add<<<1, N>>>(d_a, N);
    cudaDeviceSynchronize();

    int array[N];
    cudaMemcpy(array, d_a, N * sizeof(int), cudaMemcpyDeviceToHost);

    print_list(array, N);

    return 0;
}