#include <stdio.h>
#include <cuda.h>
#include <math.h>

void save_array(const char *filename, int *arr, int length)
{
    FILE *file = fopen(filename, "wb");
    if (file == NULL)
    {
        printf("Error opening file %s for writing.\n", filename);
        return;
    }
    fwrite(arr, sizeof(int), length, file);
    fclose(file);
}

int *read_array(const char *filename, int length)
{
    FILE *file = fopen(filename, "rb");
    if (file == NULL)
    {
        printf("Error opening file %s for reading.\n", filename);
        return NULL;
    }
    fseek(file, 0, SEEK_END);
    length = ftell(file) / sizeof(int);
    rewind(file);
    int *arr = (int *)malloc((length) * sizeof(int));

    if (arr == NULL)
    {
        printf("Memory allocation failed.\n");
        fclose(file);
        return NULL;
    }
    fread(arr, sizeof(int), length, file);
    fclose(file);

    return arr;
}

__global__ void add_chain(int *z, int *x, int *y)
{
    unsigned int id = threadIdx.x;
    z[id] = pow(x[id], 2) + pow(y[id], 3);
}

void print_list(int *array, int a_length)
{
    for (int i = 0; i < a_length; i++)
        printf("%d \t", array[i]);
    printf("\n");
}

int main()
{
    int N = 100;

    int *x = (int *)malloc(N * sizeof(int));
    int *y = (int *)malloc(N * sizeof(int));
    for (int i = 0; i < N; i++)
    {
        x[i] = i % (N / 10);
        y[i] = i % (N / 20);
    }
    save_array("x.bin", x, N);
    save_array("y.bin", y, N);

    int *new_x = read_array("x.bin", N);
    int *new_y = read_array("y.bin", N);

    int z[N], *d_z, *d_y, *d_x;
    cudaMalloc(&d_x, N * sizeof(int));
    cudaMalloc(&d_y, N * sizeof(int));
    cudaMalloc(&d_z, N * sizeof(int));

    cudaMemcpy(d_x, new_x, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, new_y, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_z, z, N * sizeof(int), cudaMemcpyHostToDevice);

    add_chain<<<1, N>>>(d_z, d_x, d_y);
    cudaDeviceSynchronize();
    cudaMemcpy(x, d_x, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(y, d_y, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(z, d_z, N * sizeof(int), cudaMemcpyDeviceToHost);
    print_list(x, N);
    print_list(y, N);
    print_list(z, N);

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
    free(x);
    free(y);
    free(new_x);
    free(new_y);
    return 0;
}