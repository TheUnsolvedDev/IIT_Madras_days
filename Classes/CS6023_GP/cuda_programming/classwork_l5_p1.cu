#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define N 10
#define M 6

__global__ void one_dim_kernel(uint16_t *matrix)
{
    uint16_t id = blockIdx.x * blockDim.x + threadIdx.x;
    matrix[id] = id;
}

int main(int argc, char **argv)
{
    uint16_t *gpu_mat;
    cudaMalloc(&gpu_mat, N * M * sizeof(uint16_t));
    uint16_t *host_mat = (uint16_t *)malloc(N * M * sizeof(uint16_t));

    one_dim_kernel<<<N, M>>>(gpu_mat);
    cudaMemcpy(host_mat, gpu_mat, N * M * sizeof(uint16_t), cudaMemcpyDeviceToHost);

    for (uint16_t i = 0; i < N; i++)
    {
        for (uint16_t j = 0; j < M; j++)
        {
            printf("%2d ", host_mat[i * M + j]);
        }
        printf("\n");
    }

    cudaFree(gpu_mat);
    free(host_mat);
    return 0;
}