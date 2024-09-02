import numpy as np
import random as rd
from objectiveFunction import get_info
from initialize import init
from plot import plot_results

class E_WOA:
    def __init__(self, pop_size, function_name, max_iterater):
        self.pop_size = pop_size
        self.function_name = function_name
        self.max_iter = max_iterater
    
    #Tính toán fitness
    def calculate_fitness(solutions, fitness_function):
        return np.array([fitness_function(solution) for solution in solutions])
    
    
    #ràng buộc phạm vi tìm kiếm
    def reset_constant(solutions, lb, ub):
        for i in range(len(solutions)):
            for j in range(len(solutions[i])):
                if solutions[i][j] < ub: solutions[i][j] = ub
                if solutions[i][j] > lb: solutions[i][j] = lb

    #Lấy ra giải pháp tốt nhất trong các giải pháp
    def get_best_solution(solutions, fitness):
        best_index = np.argmin(fitness)
        return solutions[best_index], fitness[best_index]


    def calculate_A(a1, dim):
        rand = np.random.uniform(0, 1, dim)
        return 2 * np.multiply(rand, a1) - a1

    def calculate_C(dim):
        rand = np.random.uniform(0, 1, dim)
        return 2 * rand
    
    def Encircling(best_sol, sol, A, dim):
        C = E_WOA.calculate_C(dim)
        D = np.abs(np.multiply(C, best_sol) - sol)
        return best_sol - np.multiply(A, D)
        
    #Chọn ra con ngẫu nhiên trong đàn
    def rand_solution(solutions, population_size):
        return solutions[rd.randint(0, population_size - 1)]

    #Hành động tìm kiếm dựa trên con ngẫu nhiên trong đàn
    def Searching(rand_sol, sol, A, dim):
        C = E_WOA.calculate_C(dim)
        D = np.abs(np.multiply(C, rand_sol) - sol)
        return rand_sol - np.multiply(A, D)
    
    #Hành động tấn công bong bóng dựa trên con tốt nhất trong đàn
    def Attacking(best_sol, sol, dim, a2):
        b = 1
        L = np.multiply((a2 - 1), np.random.uniform(0, 1, dim)) + 1
        D = np.linalg.norm(best_sol - sol)
        return np.multiply(np.multiply(D, np.exp(b * L)), np.cos(2.0 * np.pi * L)) + best_sol
    
    #Cập nhật fitness của mỗi solutions sau mỗi lần lặp
    def update_fitness(solutions, fitness_func, fitness, best_sol):
        fitness_list = E_WOA.calculate_fitness(solutions, fitness_func)
        best_sol_t, fitness_t = E_WOA.get_best_solution(solutions, fitness_list)
        if fitness_t < fitness:
            return fitness_t, best_sol_t
        return fitness, best_sol  
    
    def e_woa_handle(function_name, pop_size, max_iterater):
        # Lấy dữ liệu hàm test
        fitness_func, ub, lb, dim = get_info(function_name)
        # Khởi tạo quần thể
        solutions = init.iniliazation(pop_size, dim, lb, ub)
        # Tạo mảng chứa các giải pháp tốt nhất
        best_solution = []; fitness = np.inf
        best_fitness = []
        
        iter = 0
        while iter < max_iterater: 
            fitness, best_solution = E_WOA.update_fitness(solutions, fitness_func, fitness, best_solution)
            best_fitness.append(fitness)
            a1 = 2*(1 - iter / max_iterater) # a1[2, 0] 
            a2 = -1 - (1 / max_iterater * iter) # a2[-1 -2]
            belta = 1 - iter/max_iterater #belta thay cho 1 
            i = 0
            new_solutions = np.zeros_like(solutions)
            for solution in solutions:
                p = np.random.uniform(0, 1)
                if p < 0.5:
                    A = E_WOA.calculate_A(a1, dim)
                    if np.linalg.norm(A) < belta:
                        new_solutions[i, :] = E_WOA.Encircling(best_solution, solution, A, dim)
                    else:
                        random_solution = E_WOA.rand_solution(solutions, pop_size)
                        new_solutions[i, :] = E_WOA.Searching(random_solution, solution, A, dim)
                else:
                    new_solutions[i, :] = E_WOA.Attacking(best_solution, solution, dim, a2)
                i += 1
            solutions = new_solutions
            E_WOA.reset_constant(solutions, lb, ub)
            iter += 1
        plot_results(max_iterater, best_fitness)
        return best_solution, fitness    