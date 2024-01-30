#include <stdio.h>
#include <math.h>
#include <cuda.h>

#define MAX(x, y) (((x) > (y)) ? (x) : (y))

__global__ void trm()
{
    printf("%d %d %d\t", threadIdx.x, threadIdx.y, blockIdx.x);
}

__host__ void print_list(int *a, int N)
{
    for (int i = 0; i < N; i++)
    {
        printf("%d\t", a[i]);
        if ((i + 1) % (int)sqrt(N) == 0)
            printf("\n");
    }
    printf("\n");
}

__host__ void max_cpu(int *a, int *b, int *c, int N)
{
    for (int i = 0; i < N; i++)
    {
        c[i] = MAX(a[i], b[i]);
    }
}

__host__ void check_correct(int *a, int *b, int N)
{
    for (int i = 0; i < N; i++)
    {
        if (a[i] != b[i])
        {
            printf("Failed!\n");
            return;
        }
    }
    printf("Passed\n");
}

__global__ void max_gpu(int *a, int *b, int *c, int N)
{

    int idx = (threadIdx.x * blockDim.x + threadIdx.y) * gridDim.x + blockIdx.x;
    if (idx <= N)
    {
        c[idx] = MAX(a[idx], b[idx]);
        // printf(" %d %d \t", c[idx], MAX(a[idx], b[idx]));
    }
}

__host__ void hadamard_quad_cpu(int *a, int *b, int *d, int N)
{
    int d_size = 4 * N * N;
    int quad1 = 0, quad2 = N, quad3 = 2 * N * N, quad4 = N + (2 * N * N);
    for (int i = 0; i < N * N; i++)
    {
        d[i + quad1] = i;
        d[i + quad2] = i;
        d[i + quad3] = i;
        d[i + quad4] = i;
        if ((i + 1) % N == 0)
        {
            quad1 += N;
            quad2 += N;
            quad3 += N;
            quad4 += N;
        }
    }
}

__global__ void hadamard_quad_gpu(int *a, int *b, int *d, int N)
{
    // int idx
}

int main()
{
    int N = 10;
    int NN = N * N;
    printf("NN: %d\n", NN);

    int *a = (int *)malloc(NN * sizeof(int));
    int *b = (int *)malloc(NN * sizeof(int));
    int *c_cpu = (int *)malloc(NN * sizeof(int));
    int *c_gpu = (int *)malloc(NN * sizeof(int));
    int *d = (int *)malloc(4 * NN * sizeof(int));
    int *e_cpu = (int *)malloc(4 * NN * sizeof(int));
    int *e_gpu = (int *)malloc(4 * NN * sizeof(int));

    for (int i = 0; i < NN; i++)
    {
        a[i] = (((i + 1) * 2) - ((i + 1) % 2)) % N;
        b[i] = (((i + 1) * 2) - (i % 2)) % N;
        c_cpu[i] = 0;
        c_gpu[i] = 0;
    }
    for (int i = 0; i < 4 * NN; i++)
    {
        d[i] = 0;
        e_cpu[i] = 0;
        e_gpu[i] = 0;
    }

    max_cpu(a, b, c_cpu, NN);

    int *da, *db, *dc_gpu, *dd, *de_gpu;
    cudaMalloc(&da, NN * sizeof(int));
    cudaMalloc(&db, NN * sizeof(int));
    cudaMalloc(&dc_gpu, NN * sizeof(int));
    cudaMalloc(&dd, 4 * NN * sizeof(int));
    cudaMalloc(&de_gpu, 4 * NN * sizeof(int));

    cudaMemcpy(da, a, NN * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(db, b, NN * sizeof(int), cudaMemcpyHostToDevice);

    dim3 threads(32, 32, 1);
    dim3 blocks(ceil(NN / 1024.0), 1, 1);
    max_gpu<<<blocks, threads>>>(da, db, dc_gpu, NN);
    cudaDeviceSynchronize();

    cudaMemcpy(c_gpu, dc_gpu, NN * sizeof(int), cudaMemcpyDeviceToHost);
    print_list(c_cpu, NN);
    check_correct(c_cpu, c_gpu, NN);

    // threads = dim3(32, 32, 1);
    // blocks = dim3(ceil(2 * N / 32.0), ceil(2 * N / 32.0), 1);
    hadamard_quad_cpu(a, d, e_cpu, N);
    print_list(e_cpu, 4 * NN);

    cudaFree(da);
    cudaFree(db);
    cudaFree(dc_gpu);
    cudaFree(dd);
    cudaFree(de_gpu);

    free(a);
    free(b);
    free(c_cpu);
    free(c_gpu);
    free(d);
    free(e_cpu);
    free(e_gpu);

    printf("\n");
    return 0;
}