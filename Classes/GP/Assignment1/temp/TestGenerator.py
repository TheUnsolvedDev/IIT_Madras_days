'''
Test Generator for GPU A1

This script generates a test file for the GPU A1 assignment. 

Usage:

    python TestGenerator.py -f test.txt -n 1024
    where -f denotes the path of the file to generate and -n denotes the number N.
    Default values are test.txt and 1024 respectively.
'''

import random
import argparse

parser = argparse.ArgumentParser(description="Generate a test file.")

# Add the arguments
parser.add_argument('-f', '--file', type=str, default='test.txt', help='The name of the file to generate.')
parser.add_argument('-n', '--number', type=int, default=1024, help='The number N.')

# Parse the arguments
args = parser.parse_args()

filename = args.file
N = args.number

file = open(filename, "w")

####

print(N)

file.write(str(N) + "\n")

for _ in range(N):
    for _ in range(N):
        file.write(str(random.randint(0, 255)) + " ")
    file.write("\n")
print("First matrix done")

for _ in range(N):
    for _ in range(N):
        file.write(str(random.randint(0, 255)) + " ")
    file.write("\n")
print("Second Matrix done")

for _ in range(N):
    for _ in range(N):
        file.write(str(random.randint(0, 255)) + " ")
    file.write("\n")
print("Third Matrix done")

for _ in range(2 * N):
    for _ in range(2 * N):
        file.write(str(random.randint(0, 255)) + " ")
    file.write("\n")
print("Final matrix done")
