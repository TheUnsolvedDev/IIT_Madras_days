#include <stdio.h>
#include <cuda.h>
#include <strings.h>

__global__ void assign(char *array, int a_length)
{
    unsigned int id = threadIdx.x;
    if (id < a_length)
    {
        ++array[id];
    }
}


int main()
{
    char cpu_arr[] = "akdjdjaskda";
    char *gpu_arr;
    cudaMalloc(&gpu_arr, (1 + strlen(cpu_arr)) * sizeof(char));
    cudaMemcpy(gpu_arr, cpu_arr, (1 + strlen(cpu_arr)) * sizeof(char), cudaMemcpyHostToDevice);

    assign<<<1, 32>>>(gpu_arr, strlen(cpu_arr));
    cudaDeviceSynchronize();
    cudaMemcpy(cpu_arr, gpu_arr, (1 + strlen(cpu_arr)) * sizeof(char), cudaMemcpyDeviceToHost);
    printf("%s \n",cpu_arr);

    cudaFree(gpu_arr);

    return 0;
}