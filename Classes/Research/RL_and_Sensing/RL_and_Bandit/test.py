import numpy as np

size = 3
a = np.zeros((size,size))

list_of_numbers = []

for i in range(size*size-1):
    for j in range(i+1,size*size):
        list_of_numbers.append((i,j))
        print((i,j), end=" ")
    print()
    
def tx_rx_to_int(tuple,size):
    tuple_size = size*size
    return tuple[0]*tuple_size + tuple[1]

def int_to_tx_rx(int,size):
    tuple_size = size*size
    return (int//tuple_size,int%tuple_size)

action_to_index_set = {}
index_to_action_set = {}

for i in range(len(list_of_numbers)):
    action_to_index_set[list_of_numbers[i]] = i
    index_to_action_set[i] = list_of_numbers[i]
    
print(action_to_index_set)
print(index_to_action_set)