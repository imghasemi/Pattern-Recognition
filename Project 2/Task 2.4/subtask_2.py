# Subtask 2: Finding vector phi
"""
@author: Valeriia Volkovaia
"""

import itertools

# function, which returns vector phi
def subtask_2(X):
    res = []
    for L in range(0, len(X) + 1):
        for subset in itertools.combinations(X, L):
            res.append(subset) #all combinations of sets for all elements

    res[0] = (1,) #add element for empty set
    return res

if __name__ == "__main__":
    X = ['x1', 'x2', 'x3'] # some values for check
    phi = subtask_2(X) # vector phi
    print(phi)