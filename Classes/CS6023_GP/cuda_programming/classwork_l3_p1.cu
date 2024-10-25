#include <stdio.h>
#include <cuda.h>

__global__ void assign(int *array, int a_length)
{
    unsigned int id = threadIdx.x;
    if (id < a_length)
    {
        array[id] = 0;
    }
}

__global__ void add(int *array, int a_length)
{
    unsigned id_x = threadIdx.x;
    if (id_x < a_length)
    {
        array[id_x] += id_x;
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
    int *d_a, N = 1000;

    cudaMalloc(&d_a, N * sizeof(int));

    assign<<<1, N>>>(d_a, N);
    add<<<1, N>>>(d_a, N);
    cudaDeviceSynchronize();

    int array[N];
    cudaMemcpy(array, d_a, N * sizeof(int), cudaMemcpyDeviceToHost);

    print_list(array, N);
    cudaFree(d_a);
    return 0;
}