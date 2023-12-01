#include <stdio.h>
#include <cuda.h>
#include <math.h>

void write_ints(int len)
{
    FILE *fptr = fopen("digit1.txt", "w");

    for (int i = 0; i < len; i++)
    {
        fprintf(fptr, "%d\n", i);
    }
    fclose(fptr);
}

int *read_ints(int len)
{
    FILE *fp = fopen("digit1.txt", "r");
    if (fp == NULL)
    {
        printf("Error opening file.\n");
        exit(0);
    }

    int *digits = (int *)malloc(len * sizeof(int));

    int i = 0;
    while (fscanf(fp, "%d", &digits[i]) != EOF)
    {
        printf("%d %d\n", i, digits[i]);
        i++;
    }

    fclose(fp);
    return digits;
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
    write_ints(N);
    int *arr = read_ints(N);

    print_list(arr, N);

    int *d_z, *d_y, *d_x;

    // cudaMalloc(&d_x, N * sizeof(int));
    // cudaMalloc(&d_x, N * sizeof(int));
    // cudaMalloc(&d_x, N * sizeof(int));

    free(arr);
    return 0;
}