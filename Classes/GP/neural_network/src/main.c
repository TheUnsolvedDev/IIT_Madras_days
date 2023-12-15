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
    int layer_structure[3] = {10, 5, 6};
    random_weights layer1 = allocate_uniform_weights(layer_structure[0], layer_structure[1]);
    random_weights layer2 = allocate_one_weights(layer_structure[1], layer_structure[2]);

    layer1 = sigmoid_activation(layer1);
    layer2 = sigmoid_activation(layer2);

    print_weights(layer1);
    print_weights(layer2);

    free_weights(layer1);
    free_weights(layer2);
    return 0;
}