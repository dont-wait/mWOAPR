import numpy as np
import math

def F1(solution):
    return np.sum(solution**2)

def F2(solution):
    return sum(math.sin(x) for x in solution)

def F3(solution):
    return 4 * (solution[0]**2) - 2.1 * (solution[0]**4) + (solution[0]**6) / 3 + solution[0] * solution[1] - 4 * (solution[1]**2) + 4 * (solution[1]**4)




def get_info(function_name):
    if function_name == 'F1':
        return [F1, -100, 100, 2]
    if function_name == 'F2':
        return [F2, -100, 100, 2]
    if function_name == 'F3':
        return [F3, -5, 5, 2]        
    
    
        