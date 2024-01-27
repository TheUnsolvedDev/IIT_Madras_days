#include <stdio.h>
#include <cuda.h>

__global__ void trm()
{
    printf("%d %d %d\t", threadIdx.x, threadIdx.y, blockIdx.x);
}

int main()
{
    int N = 512;
    dim3 threads(32, 32, 1);
    dim3 blocks(ceil(N / 1024.0), 1, 1);
    trm<<<blocks, threads>>>();
    cudaDeviceSynchronize();

    printf("\n");
    return 0;
}