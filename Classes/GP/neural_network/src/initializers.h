#ifndef INITOIALIZERS_H
#define INITOIALIZERS_H

typedef struct random_weights
{
    float **weight;
    int size[2];

} random_weights;

random_weights allocate_zero_weights(int rows, int cols);
random_weights allocate_one_weights(int rows, int cols);
random_weights allocate_uniform_weights(int rows, int cols);
float *convert2DTo1D(float **arr2D, int rows, int cols);
float **convert1DTo2D(float *arr1D, int rows, int cols);
void free_weights(random_weights rw);
void print_weights(random_weights rw);

#endif