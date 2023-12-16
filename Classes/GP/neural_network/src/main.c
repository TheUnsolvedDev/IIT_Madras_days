#include <stdio.h>
#include <stdlib.h>

#include "initializers.h"
#include "activations.cuh"
#include "matmul.cuh"

int main() // int argc, char **argv)
{
    // initialization_present();
    // activation_present();
    // matmul_present();

    int layer_structure[3] = {17, 9, 9};
    tensor layer1 = allocate_one_weights(layer_structure[0], layer_structure[1]);
    layer1 = relu_activation(layer1);
    tensor layer2 = allocate_one_weights(layer_structure[1], layer_structure[2]);
    layer2 = relu_activation(layer2);
    tensor out = matrix_multiply(layer1, layer2);

    print_weights(layer1);
    print_weights(layer2);
    print_weights(out);

    free_weights(layer1);
    free_weights(layer2);
    free_weights(out);
    return 0;
}