#ifndef INITOIALIZERS_H

#define INITOIALIZERS_H
#define M_PI 3.14159265358979323846

typedef struct tensor
{
    float **matrix;
    int size[2];

} tensor;

tensor allocate_zero_weights(int rows, int cols);
tensor allocate_one_weights(int rows, int cols);
tensor allocate_uniform_weights(int rows, int cols);
tensor allocate_normal_weights(int rows, int cols);

float *convert2DTo1D(float **arr2D, int rows, int cols);
float **convert1DTo2D(float *arr1D, int rows, int cols);

void free_weights(tensor rw);
void print_weights(tensor rw);

#endif