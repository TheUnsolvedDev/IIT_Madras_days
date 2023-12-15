#include <stdlib.h>
#include <stdio.h>

#include "initializers.h"

void initialization_present()
{
    printf("Initialization Present\n");
}

random_weights allocate_zero_weights(int rows, int cols)
{
    random_weights rw;
    rw.size[0] = rows;
    rw.size[1] = cols;

    rw.weight = (float **)malloc(rows * sizeof(float *));
    if (rw.weight == NULL)
    {
        perror("Memory allocation failed");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < rows; i++)
    {
        rw.weight[i] = (float *)malloc(cols * sizeof(float));
        if (rw.weight[i] == NULL)
        {
            perror("Memory allocation failed");
            exit(EXIT_FAILURE);
        }
    }
    return rw;
}

random_weights allocate_one_weights(int rows, int cols)
{
    random_weights rw;
    rw.size[0] = rows;
    rw.size[1] = cols;

    rw.weight = (float **)malloc(rows * sizeof(float *));
    if (rw.weight == NULL)
    {
        perror("Memory allocation failed");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < rows; i++)
    {
        rw.weight[i] = (float *)malloc(cols * sizeof(float));
        if (rw.weight[i] == NULL)
        {
            perror("Memory allocation failed");
            exit(EXIT_FAILURE);
        }
        else
        {
            for (int j = 0; j < cols; j++)
            {
                rw.weight[i][j] = (float)1.0;
            }
        }
    }
    return rw;
}

random_weights allocate_uniform_weights(int rows, int cols)
{
    random_weights rw;
    rw.size[0] = rows;
    rw.size[1] = cols;

    rw.weight = (float **)malloc(rows * sizeof(float *));
    if (rw.weight == NULL)
    {
        perror("Memory allocation failed");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < rows; i++)
    {
        rw.weight[i] = (float *)malloc(cols * sizeof(float));
        if (rw.weight[i] == NULL)
        {
            perror("Memory allocation failed");
            exit(EXIT_FAILURE);
        }
        else
        {
            for (int j = 0; j < cols; j++)
            {
                rw.weight[i][j] = (float)(rand() % 10000) / 10000.0;
            }
        }
    }
    return rw;
}

void free_weights(random_weights rw)
{
    for (int i = 0; i < rw.size[0]; i++)
    {
        free(rw.weight[i]);
    }
    free(rw.weight);
}

void print_weights(random_weights rw)
{
    printf("\nweights:\n");
    for (int i = 0; i < rw.size[0]; i++)
    {
        for (int j = 0; j < rw.size[1]; j++)
        {
            printf("%.4f\t", rw.weight[i][j]);
        }
        printf("\n");
    }
    printf("shape:(%d,%d)\n", rw.size[0], rw.size[1]);
}

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