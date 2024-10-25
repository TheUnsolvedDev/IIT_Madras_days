#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <limits.h>

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

/*
Strategy 1:
- Let's fix num_threads = 256; so 256 data will be handled per block
suppose for 1000 distances we need 4 blocks (3x256 + 1x232)

sqrt_distance<<<num_blocks,num_threads>>>()


*/

__global__ void sqrt_distance(unsigned int *x, unsigned int *y, float *res, int n)
{
    unsigned int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int idx_x = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx_x < n && idx_y < n && idx_x < idx_y)
    {
        float sub_dist_x = pow((float)x[idx_x] - (float)x[idx_y], 2);
        float sub_dist_y = pow((float)y[idx_x] - (float)y[idx_y], 2);
        int index = (int)((idx_y * (idx_y - 1)) / 2) + idx_x;
        float distance = sqrt(sub_dist_x + sub_dist_y);
        res[index] = distance;
    }
}

__global__ void maximum_distance(float *res, int length, float *maxm)
{
    unsigned int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx_x < length && idx_y < length && idx_x < idx_y)
    {
        printf("%d %d \t", idx_x, idx_y);
        float maximum = INT_MIN;
        maximum = MAX(MAX(res[idx_x], res[idx_y]), maximum);
        *maxm = maximum;
    }
}

__global__
void find_max(int max_x, int max_y, float *tot, float *x, float *y)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    if(i < max_x && j<max_y) {
        if(*tot < x[i])
            atomicExch(tot, x[i]);
    }
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
    // int num_points = atoi(argv[1]);
    int num_points;
    scanf("%d", &num_points);
    printf("%d total number of points\n", num_points);

    unsigned int *x = (unsigned int *)malloc(num_points * sizeof(unsigned int));
    unsigned int *y = (unsigned int *)malloc(num_points * sizeof(unsigned int));

    for (int i = 0; i < num_points; i++)
    {
        scanf("%u %u", &x[i], &y[i]);
    }

    printf("\nThe coordinates are:\n");
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

    int num_threads = 16;
    int num_blocks_for_distance = ceil(float(num_points) / (float)num_threads);
    int num_blocks_for_maximum = ceil(float(cross_points / 2) / (float)num_threads);

    dim3 grid_for_distance(num_blocks_for_distance, num_blocks_for_distance);
    dim3 grid_for_maximum(num_blocks_for_maximum, num_blocks_for_maximum);
    dim3 block(num_threads, num_threads);

    sqrt_distance<<<grid_for_distance, block>>>(dx, dy, dsqrt_dist, num_points);
    maximum_distance<<<grid_for_maximum, block>>>(dsqrt_dist, cross_points / 2, dmax_dist);
    cudaDeviceSynchronize();

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