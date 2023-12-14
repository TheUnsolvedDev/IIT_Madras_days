#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define BLOCKSIZE 1024

__global__ void one_dim_kernel(unsigned int *vector, unsigned int vector_size)
{
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < vector_size)
        vector[id] = id;
}

void print_list(unsigned int *array, int a_length)
{
    for (int i = 0; i < a_length; i++)
        printf("%d \t", array[i]);
    printf("\n");
}

int main(int argc, char **argv)
{
    unsigned int N = atoi(argv[1]);
    unsigned int *gpu_vector, *host_vector;
    cudaMalloc(&gpu_vector, N * sizeof(unsigned int));
    host_vector = (unsigned int *)malloc(N * sizeof(unsigned int));

    unsigned int nblocks = ceil((float)N / BLOCKSIZE);
    printf("Number of blocks: %d \n", nblocks);

    one_dim_kernel<<<nblocks, BLOCKSIZE>>>(gpu_vector, N);
    cudaDeviceSynchronize();

    cudaMemcpy(host_vector, gpu_vector, N * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    print_list(host_vector, N);

    free(host_vector);
    cudaFree(gpu_vector);
    return 0;
}