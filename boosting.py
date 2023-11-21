import numpy as np

def rectangle_filter1(rows, cols):
    lhs = np.ones((rows,cols))
    rhs = np.ones((rows,cols)) * -1
    return np.concatenate((lhs, rhs), axis=1)

def rectangle_filter2(rows, cols):
    top = np.ones((rows,cols))
    bottom = np.ones((rows,cols)) * -1
    return np.concatenate((top, bottom), axis=0)

def rectangle_filter3(rows, cols):
    lhs = np.ones((rows,cols))
    middle = np.ones((rows,cols)) * -2
    rhs = np.ones((rows,cols))
    return np.concatenate((lhs, middle, rhs), axis=1)

def rectangle_filter4(rows, cols):
    top = np.ones((rows,cols))
    middle = np.ones((rows,cols)) * -2
    bottom = np.ones((rows,cols))
    return np.concatenate((top, middle, bottom), axis=0)

def rectangle_filter5(rows, cols):
    pos = np.ones((rows,cols))
    neg = np.ones((rows,cols)) * -1
    top = np.concatenate((pos, neg), axis=1)
    bottom = np.concatenate((neg, pos), axis=1)
    return np.concatenate((top, bottom), axis=0)