import numpy as np
import tqdm


def print_matrix_to_file(matrices, file_path):
    with open(file_path, 'w') as file:
        file.write(str(matrices[0])+'\n')
        for matrix in matrices[1:]:
            for row in matrix:
                file.write(' '.join(map(str, row)) + '\n')


def create_matrix(N=2):
    A = np.random.randint(low=1, high=255, size=(N, N))
    B = np.random.randint(low=1, high=255, size=(N, N))
    C = np.random.randint(low=1, high=255, size=(N, N))
    D = np.random.randint(low=1, high=255, size=(2*N, 2*N))
    return (N, A, B, C, D)


if __name__ == '__main__':
    for i in tqdm.tqdm(range(15, 21)):
        file_name = f"test{i}.txt"
        N = np.random.randint(low=2**14, high=2**16)
        print_matrix_to_file(create_matrix(N), file_name)
