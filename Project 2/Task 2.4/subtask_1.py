# Subtask 1: Optimizing vector w for cellular automata, using X as-is
"""
@author: Valeriia Volkovaia
"""

import numpy as np

#least squares from numpy
def linear_regression(x,y):
    X = np.column_stack(x + [[1] * len(x[0])])
    w = np.linalg.lstsq(X, y)[0] #vector w
    Y = np.dot(X, w) #predicted values
    return w, Y

if __name__ == "__main__":
    rule110_y = [-1, 1, 1, 1, -1, 1, 1, -1] #y for rule 110
    rule126_y = [-1, 1, 1, 1, 1, 1, 1, -1] #y for rule 126
    X = [[1, 1, 1, 1, -1, -1, -1, -1],
         [1, 1, -1, -1, 1, 1, -1, -1],
         [1, -1, 1, -1, 1, -1, 1, -1],
         ] # X is the same for the both rules
    [w_110, y_110] = linear_regression(X, rule110_y)
    [w_126, y_126] = linear_regression(X, rule126_y)
    print("Vector w for rule 110:")
    print(w_110)
    print("Predicted values for rule 110: ")
    print(y_110)
    print("Vector w for rule 126:")
    print(w_126)
    print("Predicted values for rule 126: ")
    print(y_126)