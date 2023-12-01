from strategy import *

if __name__ == "__main__":
    SIZE = 8
    ITERATION = 1000
    for i in range(4):
        strategy1(0.75, i, SIZE, ITERATION)
        strategy2(0.75, i, SIZE, ITERATION)
        strategy2(0.75, i, SIZE, ITERATION, True)
