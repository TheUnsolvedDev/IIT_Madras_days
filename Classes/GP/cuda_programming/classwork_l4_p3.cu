#include <stdio.h>
#include <cuda.h>

#define N 5
#define M 6

__global__ void two_dim_kernel(unsigned int *matrix)
{
    unsigned int id = threadIdx.x * blockDim.y + threadIdx.y;
    matrix[id] = id;
}

int main()
{
    dim3 block_new(N, M);
    unsigned int *hmatrix = (unsigned int *)malloc(N * M * sizeof(unsigned int));
    unsigned int *matrix;
    cudaMalloc(&matrix, N * M * sizeof(unsigned int));

    two_dim_kernel<<<1, block_new>>>(matrix);
    cudaMemcpy(hmatrix, matrix, N * M * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < M; j++)
        {
            printf("%2d -", hmatrix[i * M + j]);
        }
        printf("\n");
    }

    cudaFree(matrix);
    free(hmatrix);

    return 0;
}