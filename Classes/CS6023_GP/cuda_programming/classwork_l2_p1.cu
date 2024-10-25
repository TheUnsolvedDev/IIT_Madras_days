#include <stdio.h>
#include <cuda.h>

#define N 10

__global__ void print_square()
{
    for (int i = 0; i < N; i++)
    {
        printf("%d \t", i * i);
    }
    printf("\n");
}

__global__ void thread_print()
{
    printf("%d \n", threadIdx.x);
}

int main()
{
    print_square<<<1, 1>>>();
    thread_print<<<1, N>>>();
    cudaDeviceSynchronize();
    return 0;
}