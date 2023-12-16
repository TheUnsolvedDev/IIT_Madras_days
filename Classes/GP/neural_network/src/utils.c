#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "utils.h"

float *convert2DTo1D(float **arr2D, int rows, int cols)
{
    float *arr1D = (float *)malloc(rows * cols * sizeof(float));
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            arr1D[i * cols + j] = arr2D[i][j];
        }
    }
    for (int i = 0; i < rows; i++)
    {
        free(arr2D[i]);
    }
    free(arr2D);
    return arr1D;
}

float **convert1DTo2D(float *arr1D, int rows, int cols)
{
    float **arr2D = (float **)malloc(rows * sizeof(float *));
    for (int i = 0; i < rows; i++)
    {
        arr2D[i] = (float *)malloc(cols * sizeof(float));
        for (int j = 0; j < cols; j++)
        {
            arr2D[i][j] = arr1D[i * cols + j];
        }
    }
    free(arr1D);
    return arr2D;
}