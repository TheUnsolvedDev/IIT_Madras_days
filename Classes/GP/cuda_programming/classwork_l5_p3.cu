#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <limits.h>

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

__global__ void sqrt_distance(unsigned int *x, unsigned int *y, float *res, int n)
{
    unsigned int i = threadIdx.x;
    unsigned int j = threadIdx.y;

    if (threadIdx.x < threadIdx.y)
    {
        float sub_dist_x = pow((float)x[i] - (float)x[j], 2);
        float sub_dist_y = pow((float)y[i] - (float)y[j], 2);
        int index = (int)((j * (j - 1)) / 2) + i;
        float distance = sqrt(sub_dist_x + sub_dist_y);
        res[index] = distance;
    }
}

__global__ void maximum_distance(float *res, int length, float *maxm)
{
    float maximum = INT_MIN;
    maximum = MAX(MAX(res[threadIdx.x], res[threadIdx.y]), maximum);
    *maxm = maximum;
}

void print_list(unsigned int *array, int a_length)
{
    for (int i = 0; i < a_length; i++)
        printf("%d \t", array[i]);
    printf("\n");
}

void print_list_float(float *array, int a_length)
{
    for (int i = 0; i < a_length; i++)
        printf("%10.3f \t", array[i]);
    printf("\n");
}

int main(int argc, char **argv)
{
    int num_points = atoi(argv[1]);
    printf("%d total number of points\n", num_points);
    if (num_points > 32)
    {
        printf("The code wasn't supposed to work better for this sorry!!\n");
        exit(0);
    }

    unsigned int *x = (unsigned int *)malloc(num_points * sizeof(unsigned int));
    unsigned int *y = (unsigned int *)malloc(num_points * sizeof(unsigned int));

    for (int i = 0; i < num_points; i++)
    {
        scanf("%u %u", &x[i], &y[i]);
    }

    printf("The coordinates are:\n");
    print_list(x, num_points);
    print_list(y, num_points);

    unsigned int *dx, *dy;
    int cross_points = (num_points) * (num_points - 1);
    float *sqrt_dist = (float *)malloc(cross_points * sizeof(float));
    float *dsqrt_dist, *dmax_dist, *max_dist = (float *)malloc(sizeof(float));

    cudaMalloc(&dx, num_points * sizeof(unsigned int));
    cudaMalloc(&dy, num_points * sizeof(unsigned int));
    cudaMalloc(&dsqrt_dist, cross_points / 2 * sizeof(float));
    cudaMalloc(&dmax_dist, sizeof(float));

    cudaMemcpy(dy, y, num_points * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(dx, x, num_points * sizeof(unsigned int), cudaMemcpyHostToDevice);

    dim3 block(num_points, num_points);
    sqrt_distance<<<1, block>>>(dx, dy, dsqrt_dist, num_points);
    maximum_distance<<<1, block>>>(dsqrt_dist, cross_points / 2, dmax_dist);

    cudaMemcpy(sqrt_dist, dsqrt_dist, cross_points / 2 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(max_dist, dmax_dist, sizeof(float), cudaMemcpyDeviceToHost);

    printf("The squared distances are:\n");
    print_list_float(sqrt_dist, cross_points / 2);

    printf("The maximum distance is ");
    print_list_float(max_dist, 1);

    cudaFree(dx);
    cudaFree(dy);
    cudaFree(dsqrt_dist);
    cudaFree(dmax_dist);
    free(x);
    free(y);
    free(sqrt_dist);
    free(max_dist);
}