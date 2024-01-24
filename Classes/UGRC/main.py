from strategy import *
import pickle

no_reg = []
reg = []
zero_prior = []


def write_value(value):
    with open('maps.pkl', 'wb') as f:
        pickle.dump(value, f)


def read_value():
    with open('maps.pkl', 'rb') as f:
        val = pickle.load(f)
    return val


if __name__ == "__main__":
    SIZE = 10
    NUM_MAPS = 8
    ITERATION = 1000
    maps = {}
    for i in range(NUM_MAPS):
        true, pred = strategy1(0.6, i, SIZE, ITERATION)
        no_reg.append((true, pred))
        true, pred = strategy2(0.6, i, SIZE, ITERATION)
        reg.append((true, pred))
        true, pred = strategy2(0.6, i, SIZE, ITERATION, True)
        zero_prior.append((true, pred))

    maps['no_reg'] = no_reg
    maps['reg'] = reg
    maps['zero_prior'] = zero_prior
    write_value(maps)
    print(read_value())