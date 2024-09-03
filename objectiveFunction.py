import numpy as np
import math

def F1(solution):
    return np.sum(solution**2)

def F2(solution):
    return sum(math.sin(x) for x in solution)

def F3(solution):
    return 4 * (solution[0]**2) - 2.1 * (solution[0]**4) + (solution[0]**6) / 3 + solution[0] * solution[1] - 4 * (solution[1]**2) + 4 * (solution[1]**4)


def F10(solution):
    n = len(solution)
    sum_x2 = np.sum(solution**2)
    sum_cos = np.sum(np.cos(2 * np.pi * solution))
    
    term1 = -20 * np.exp(-0.2 * np.sqrt(1/n * sum_x2))
    term2 = np.exp(1/n * sum_cos)
    
    return term1 - term2 + 20 + np.e


def get_info(function_name):
    if function_name == 'F1':
        return [F1, -100, 100, 2]
    if function_name == 'F2':
        return [F2, -100, 100, 2]
    if function_name == 'F3':
        return [F3, -5, 5, 2]        
    if function_name == 'F10':
        return [F10, -32, 32, 30]    
    
        