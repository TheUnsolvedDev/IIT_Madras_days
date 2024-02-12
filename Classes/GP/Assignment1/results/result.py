import numpy as np
import tqdm


def find_hadamard_product(A, B):
    return np.multiply(A, B.T)


def find_weight_matrix(A, C):
    return np.maximum(C, A)


def find_final_matrix(A, D, N):
    F = np.zeros((2 * N, 2 * N), dtype=int)
    F[:N, :N] = np.multiply(D[:N, :N], A)  # Quadrant D1
    F[:N, N:] = np.multiply(D[:N, N:], A)  # Quadrant D2
    F[N:, :N] = np.multiply(D[N:, :N], A)  # Quadrant D3
    F[N:, N:] = np.multiply(D[N:, N:], A)  # Quadrant D4
    return F


def print_matrix_to_file(matrix, file_path):
    with open(file_path, 'w') as file:
        for row in matrix:
            file.write(' '.join(map(str, row)) + ' \n')


def main(file_name='test1.txt'):
    with open(file_name, 'r') as file:
        N = int(file.readline())

        A = np.zeros((N, N), dtype=int)
        B = np.zeros((N, N), dtype=int)
        C = np.zeros((N, N), dtype=int)
        D = np.zeros((2 * N, 2 * N), dtype=int)

        for i in range(N):
            A[i] = list(map(int, file.readline().split()))

        for i in range(N):
            B[i] = list(map(int, file.readline().split()))

        for i in range(N):
            C[i] = list(map(int, file.readline().split()))

        for i in range(2 * N):
            D[i] = list(map(int, file.readline().split()))

    A = find_hadamard_product(A, B)
    A = find_weight_matrix(A, C)
    F = find_final_matrix(A, D, N)
    new_file = file_name.replace('test', 'out')
    print_matrix_to_file(F, new_file)


def are_files_equal(file_path1, file_path2):
    try:
        with open(file_path1, 'r') as file1, open(file_path2, 'r') as file2:
            content1 = file1.read()
            content2 = file2.read()
            return content1 == content2
    except FileNotFoundError:
        print("One or both of the files do not exist.")
        return False


if __name__ == "__main__":
    for i in tqdm.tqdm(range(1, 16)):
        file = f"test{i}.txt"
        main(file)
