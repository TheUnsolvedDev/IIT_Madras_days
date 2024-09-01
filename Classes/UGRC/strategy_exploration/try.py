import numpy as np

def convert_actions(num: int, to_base: int) -> np.ndarray:
    """
    Converts an integer action number to a base-size representation.

    Args:
        num (int): The action number to be converted.
        to_base (int): The base to convert the action number to.

    Returns:
        np.ndarray: The converted action number with shape (4, ).
    """
    temp = np.zeros(4, dtype=np.int16)
    count = 0
    while num > 0:
        digit = num % to_base
        num //= to_base
        temp[count] = digit
        count += 1
    return temp[::-1]

for i in range(20**4-1):
    print(convert_actions(i, 20))