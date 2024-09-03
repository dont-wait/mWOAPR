import numpy as np
class init:
    def initialization(pop_size, dim, lb, ub):
        population = np.random.rand(pop_size, dim) * (ub - lb) + lb  
        return population
    