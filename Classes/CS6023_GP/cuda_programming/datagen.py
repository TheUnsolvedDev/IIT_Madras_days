import numpy as np


def write_data(num_coordinates):
    with open('input.in', 'w') as f:
        f.write(str(num_coordinates)+"\n")
        for i in range(num_coordinates):
            x = np.random.randint(low=0, high=15)
            y = np.random.randint(low=0, high=15)
            f.write(str(x)+" "+str(y)+"\n")


def calculate_max_dist():
    with open('input.in', 'r') as f:
        data = f.readlines()

    num_coordinates = int(data.pop(0).replace('\n', ''))
    locs = []
    for coordinate in data:
        loc = list(map(int, coordinate.replace('\n', '').split(' ')))
        locs.append(loc)
    locs = np.array(locs)
    for i in range(len(locs)):
        for j in range(i+1, len(locs)):
            print(locs[i], locs[j], np.sqrt(
                np.sum(np.square(locs[i] - locs[j]))))


if __name__ == '__main__':
    # write_data(6)
    calculate_max_dist()
