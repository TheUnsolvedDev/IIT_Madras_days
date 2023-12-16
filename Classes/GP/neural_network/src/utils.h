#ifndef UTILS_H

#define UTILS_H
#define NUM_THREADS 512
#define NUM_2D_THREADS 16
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

float *convert2DTo1D(float **arr2D, int rows, int cols);
float **convert1DTo2D(float *arr1D, int rows, int cols);

#endif