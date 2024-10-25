/**
 *   CS6023: GPU Programming
 *   Assignment 2
 *
 *   Please don't change any existing code in this file.
 *
 *   Please add necessary memory APIs for your implementation. Use cudaFree()
 *   to free up memory as soon as you're done with an allocation.
 *   This will ensure that you don't run out of memory while running
 *   large test cases. Use the minimum required memory for your
 *   implementation. DO NOT change the kernel configuration parameters.
 */

#include <chrono>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <cuda.h>

using namespace std;

using std::cin;
using std::cout;

typedef long long ll;

// __global__ void convolution_kernel(long int *image, long int *kernel, long int *answer, int width, int height, int kernel_size)
// {
//     unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
//     unsigned int col = blockIdx.y * blockDim.y + threadIdx.y;

//     if (row < width && col < height)
//     {
//         unsigned int kernel_width_half = kernel_size / 2;
//         float result = 0.0f;

//         for (int i = 0; i < kernel_size; ++i)
//         {
//             for (int j = 0; j < kernel_size; ++j)
//             {
//                 int img_row = row - kernel_width_half + i;
//                 int img_col = col - kernel_width_half + j;
//                 if (img_row >= 0 && img_row < width && img_col >= 0 && img_col < height)
//                 {
//                     result += image[img_row * height + img_col] * kernel[i * kernel_size + j];
//                 }
//             }
//         }
//         answer[row * height + col] = result;
//     }
// }

// __global__ void convolution_kernel(long int *image, long int *kernel, long int *answer, int width, int height, int kernel_size)
// {
//     unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
//     unsigned int col = blockIdx.y * blockDim.y + threadIdx.y;

//     extern __shared__ long int kernel_shared[];
//     unsigned int kernel_idx = threadIdx.x * kernel_size + threadIdx.y;
//     if (kernel_idx < kernel_size * kernel_size)
//     {
//         kernel_shared[kernel_idx] = kernel[kernel_idx];
//     }
//     __syncthreads();

//     if (row < width && col < height)
//     {
//         unsigned int kernel_width_half = kernel_size / 2;
//         float result = 0.0f;

//         for (int i = 0; i < kernel_size; ++i)
//         {
//             for (int j = 0; j < kernel_size; ++j)
//             {
//                 int img_row = row - kernel_width_half + i;
//                 int img_col = col - kernel_width_half + j;

//                 if (img_row >= 0 && img_row < width && img_col >= 0 && img_col < height)
//                 {
//                     result += image[img_row * height + img_col] * kernel_shared[i * kernel_size + j];
//                 }
//             }
//         }
//         answer[row * height + col] = result;
//     }
// }


__global__ void convolution_kernel(long int *image, long int *kernel, long int *answer, int width, int height, int kernel_size)
{
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int col = blockIdx.y * blockDim.y + threadIdx.y;

    extern __shared__ long int kernel_shared[], image_shared[];
    for (unsigned int kernel_idx = threadIdx.x; kernel_idx < kernel_size * kernel_size; kernel_idx += blockDim.x)
    {
        kernel_shared[kernel_idx] = kernel[kernel_idx];
    }
    __syncthreads();

    if (row < width && col < height)
    {
        unsigned int kernel_width_half = kernel_size / 2;
        long int result = 0;

        for (int i = 0; i < kernel_size; ++i)
        {
            for (int j = 0; j < kernel_size; ++j)
            {
                int img_row = row - kernel_width_half + i;
                int img_col = col - kernel_width_half + j;

                if (img_row >= 0 && img_row < width && img_col >= 0 && img_col < height)
                {
                    result += image[img_row * height + img_col] * kernel_shared[i * kernel_size + j];
                }
            }
        }
        answer[row * height + col] = result;
    }
}

int main(int argc, char **argv)
{

    int m, n, k;
    cin >> m >> n >> k;

    long int *h_mat = new long int[m * n];
    long int *h_filter = new long int[k * k];

    long int *h_ans = new long int[m * n];

    for (long int i = 0; i < m * n; i++)
    {
        cin >> h_mat[i];
    }

    for (long int i = 0; i < k * k; i++)
    {
        cin >> h_filter[i];
    }

    /**
     *
     * DO NOT CHANGE ANYTHING ABOVE THIS LINE
     *
     **/

    /****************************************************Start Here***********************************************************/
    long int *dh_mat, *dh_filter, *dh_ans;
    size_t dh_mat_size = sizeof(long int) * m * n;
    size_t dh_filter_size = sizeof(long int) * k * k;

    cudaMalloc(&dh_mat, dh_mat_size);
    cudaMalloc(&dh_filter, dh_filter_size);
    cudaMalloc(&dh_ans, dh_mat_size);

    cudaMemcpy(dh_mat, h_mat, dh_mat_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dh_filter, h_filter, dh_filter_size, cudaMemcpyHostToDevice);

    dim3 block(32, 32, 1), grid(ceil(m / (float)block.x), ceil(n / (float)block.y), 1);

    cudaFuncSetCacheConfig(convolution_kernel, cudaFuncCachePreferShared);
    auto start = std::chrono::high_resolution_clock::now(); // keep it just before the kernel launch
    convolution_kernel<<<grid, block, dh_filter_size>>>(dh_mat, dh_filter, dh_ans, m, n, k);
    auto end = std::chrono::high_resolution_clock::now(); // keep it just after the kernel launch

    cudaMemcpy(h_ans, dh_ans, dh_mat_size, cudaMemcpyDeviceToHost);

    cudaFree(dh_mat);
    cudaFree(dh_filter);
    cudaFree(dh_ans);

    /*$$$$$$$$$$$$$$$$$$$$$$$$Make sure your final output from the device is stored in h_ans.$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$*/
    std::chrono::duration<double> elapsed1 = end - start;
    /**
     *
     * DO NOT CHANGE ANYTHING BELOW THIS LINE
     *
     */

    std::ofstream file("cuda.out");
    if (file.is_open())
    {
        for (long int i = 0; i < m; i++)
        {
            for (long int j = 0; j < n; j++)
            {
                file << h_ans[i * n + j] << " ";
            }
            file << "\n";
        }
        file.close();
    }
    else
    {
        std::cout << "Unable to open file";
    }

    std::ofstream file2("cuda_timing.out");
    if (file2.is_open())
    {
        file2 << elapsed1.count() << "\n";
        file2.close();
    }
    else
    {
        std::cout << "Unable to open file";
    }

    return 0;
}