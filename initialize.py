import numpy as np
class init:
    def iniliazation(pop_size, dim, lb, ub):
        population = np.random.rand(pop_size, dim) * (ub - lb) + lb  
        return population