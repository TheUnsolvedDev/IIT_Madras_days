#include <cuda.h>
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void add(int *a, int *b, int *c)
{
	*c = *a + *b;
}

int main()
{
	int a, b, c;
	int *d_a, *d_b, *d_c;
	int size = sizeof(int);

	cudaMalloc(&d_a, size);
	cudaMalloc(&d_b, size);
	cudaMalloc(&d_c, size);

	a = 2, b = 102;

	cudaMemcpy(d_a, &a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, &b, size, cudaMemcpyHostToDevice);

	add<<<1, 1>>>(d_a, d_b, d_c);
	cudaMemcpy(&c, d_c, size, cudaMemcpyDeviceToHost);

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	printf("The result is %d\n", c);

	return 0;
}
