import numpy as np



if __name__ == '__main__':
    field = np.ones((10,10))
    division = 4
    
    field_divisions = np.array_split(field, division, axis=1)
    field_divisions = np.array_split(field_divisions, division, axis=0)
    print(field_divisions)
    np.arr