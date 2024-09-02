import numpy as np
import math

def F1(solution):
    return np.sum(solution**2)

def F2(solution):
    return sum(math.sin(x) for x in solution)

def F3(solution):
    return 4 * (solution[0]**2) - 2.1 * (solution[0]**4) + (solution[0]**6) / 3 + solution[0] * solution[1] - 4 * (solution[1]**2) + 4 * (solution[1]**4)
def F12(solution):
    def Ufun(solution, a, b, c):
        # Định nghĩa logic của Ufun
        return np.sum(solution)  # Cài đặt mẫu

    dim = solution.shape[0]
    term1 = (np.pi / dim) * (10 * (np.sin(np.pi * (1 + (solution[0] + 1) / 4)) ** 2))
    
    term2 = np.sum((((solution[:dim-1] + 1) / 4) ** 2) * (1 + 10 * (np.sin(np.pi * (1 + (solution[1:dim] + 1) / 4)) ** 2)))
    
    term3 = ((solution[dim-1] + 1) / 4) ** 2
    
    o = term1 + term2 + term3 + Ufun(solution, 10, 100, 4)
    
    return o



def get_info(function_name):
    if function_name == 'F1':
        return [F1, -100, 100, 2]
    if function_name == 'F2':
        return [F2, -100, 100, 2]
    if function_name == 'F5':
        return [F3, -5, 5, 2]        
    if function_name == 'F12':
        return [F12, -50, 50, 30]
    
        