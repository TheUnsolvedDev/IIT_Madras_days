import numpy as np
import random
import matplotlib.pyplot as plt
import torch
import jax
import jax.numpy as jnp

SEED = 3407
MU, SIGMA = 0, 0.02
AGGREGATED_MU, AGGREGATED_SIGMA = 1, 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def min_max_scaling(data):
    min_val = np.min(data)
    max_val = np.max(data)
    scaled_data = (data - min_val) / (max_val - min_val)
    return scaled_data


# def plot(array, scaled_data=True):
#     if scaled_data == True:
#         plt.imshow(min_max_scaling(array))
#     else:
#         plt.imshow(array)
#     plt.show()


# def psnr(img1, img2):
#     if img1.shape != img2.shape:
#         raise ValueError("Input images must have the same dimensions")
#     mse = np.mean((img1 - img2) ** 2)
#     dynamic_range = np.max(img1) - np.min(img1)
#     psnr_value = 20 * np.log10(dynamic_range) - 10 * np.log10(mse)
#     return psnr_value


@jax.jit
def psnr(img1, img2):
    if img1.shape != img2.shape:
        raise ValueError("Input images must have the same dimensions")
    mse = jnp.mean((img1 - img2) ** 2)
    dynamic_range = jnp.max(img1) - jnp.min(img1)
    psnr_value = 20 * jnp.log10(dynamic_range) - 10 * jnp.log10(mse)
    return psnr_value


def bresenham_line(x1, y1, x2, y2):
    line_points = []
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    step_x = 1 if x1 < x2 else -1
    step_y = 1 if y1 < y2 else -1
    error = dx - dy

    while True:
        line_points.append((x1, y1))

        if x1 == x2 and y1 == y2:
            break

        double_error = 2 * error
        if double_error > -dy:
            error -= dy
            x1 += step_x

        if double_error < dx:
            error += dx
            y1 += step_y

    return line_points


class Map:
    def __init__(self,map_size = (50,50)):
        attenuation_steel = 2
        attenuation_land = 1
        attenuation_air = 0.5

        # self.map = np.random.randint(0, 2, [10, 12]).astype(np.float32)
        self.matrix_size = map_size
        matrix = np.zeros(self.matrix_size, dtype=np.float32)
        m, n = matrix.shape
        center_row = m // 2
        center_col = n // 2
        arm_height = min(self.matrix_size)//2
        arm_width = min(self.matrix_size)//4

        matrix[center_row - arm_height//2: center_row + arm_height//2 + 1,
               center_col - arm_width//2: center_col + arm_width//2 + 1] = 1

        matrix[center_row - arm_width//2: center_row + arm_width//2 + 1,
               center_col - arm_height//2: center_col + arm_height//2 + 1] = 1
        
        matrix[center_row - arm_height//4: center_row + arm_height//4 + 1,
               center_col - arm_width//4: center_col + arm_width//4 + 1] = 2

        matrix[center_row - arm_width//4: center_row + arm_width//4 + 1,
               center_col - arm_height//4: center_col + arm_height//4 + 1] = 2

        self.map = matrix

        noise = np.random.normal(MU, SIGMA, size=self.map.shape)

        self.sides = []
        for i in range(self.map.shape[1]):
            self.sides.append([0, i])

        for i in range(1, self.map.shape[0]-1):
            self.sides.append([i, 0])

        for i in range(self.map.shape[1]):
            self.sides.append([self.map.shape[0]-1, i])

        for i in range(1, self.map.shape[0]-1):
            self.sides.append([i, self.map.shape[1]-1])

        self.attenuation_map = self.map.copy()
        self.attenuation_map[self.attenuation_map == 0] = attenuation_air
        self.attenuation_map[self.attenuation_map == 1] = attenuation_land
        self.attenuation_map[self.attenuation_map == 2] = attenuation_steel
        self.attenuation_map += noise

        # plot(self.attenuation_map)  # , scaled_data=False)

    def find_cut_indices(self, matrix, x1, y1, x2, y2):
        cut_indices = []
        line_points = bresenham_line(x1, y1, x2, y2)

        for x, y in line_points:
            if 0 <= x < len(matrix) and 0 <= y < len(matrix[0]):
                cut_indices.append((x, y))

        return cut_indices

    def attenuation_result(self):
        transmitter, reflector = random.choices(self.sides, k=2)
        while transmitter == reflector:
            transmitter, reflector = random.choices(self.sides, k=2)

        cuts = self.find_cut_indices(self.map, *transmitter, *reflector)
        indices = list(map(lambda x: x[0]*self.map.shape[0]+x[1], cuts))

        mask = np.zeros(self.map.shape)
        for i in cuts:
            mask[i[0]][i[1]] = 1

        A_vector = mask.astype(np.float32)
        B_vector = np.dot(A_vector.reshape(-1),
                          self.attenuation_map.reshape(-1, 1))  # + np.random.normal(AGGREGATED_MU, AGGREGATED_SIGMA)

        return transmitter, reflector, indices, A_vector, B_vector

    def generate_samples(self, n_dipole_samples=20000):
        A_samples_list = []
        B_samples_list = []
        tr_pair = {}

        with torch.no_grad():
            while len(A_samples_list) < n_dipole_samples:
                t, r, _, A, B = self.attenuation_result()
                if tr_pair.get((*t, *r), 'Not Found') == 'Not Found':
                    tr_pair[(*t, *r)] = 'Found'
                    A_samples_list.append(A)
                    B_samples_list.append(B)

            A_samples = np.array(A_samples_list, dtype=np.float32)
            B_samples = np.array(B_samples_list, dtype=np.float32)

            A_tensor = torch.tensor(A_samples, dtype=torch.float32, device=device).view(
                n_dipole_samples, -1)
            B_tensor = torch.tensor(B_samples, dtype=torch.float32, device=device).view(
                n_dipole_samples, -1)

            term1 = torch.pinverse(torch.mm(A_tensor.T, A_tensor))
            term2 = torch.mm(A_tensor.T, B_tensor)
            theta = torch.mm(term1, term2)
            # plot(theta.view(self.matrix_size).cpu().numpy())

        return self.attenuation_map, theta.view(self.matrix_size).cpu().numpy()


if __name__ == '__main__':
    m = Map()
    n_dipoles = [i for i in range(1000,20001,500)]
    psnr_values = []
    
    for n in n_dipoles:
        original, generated = m.generate_samples(n)
        psnr_values.append(psnr(original, generated))
        print('PSNR Value of the matrices is:', psnr_values[-1])
    print(psnr_values)