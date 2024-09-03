import numpy as np
import random as rd
from objectiveFunction import get_info
from initialize import init
from plot import plot_results, plot_solutions_update

class mWOAPR:
    def __init__(self, pop_size, function_name, max_iterater):
        self.pop_size = pop_size
        self.function_name = function_name
        self.max_iterater = max_iterater
    
    #Tính toán fitness
    def calculate_fitness(solutions, fitness_function):
        return np.array([fitness_function(solution) for solution in solutions])
    
    
    # #ràng buộc phạm vi tìm kiếm
    # def reset_constant(solutions, lb, ub):
    #     for i in range(len(solutions)):
    #         for j in range(len(solutions[i])):
    #             if solutions[i][j] < ub: solutions[i][j] = ub
    #             if solutions[i][j] > lb: solutions[i][j] = lb

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
        C = mWOAPR.calculate_C(dim)
        D = np.abs(np.multiply(C, best_sol) - sol)
        return best_sol - np.multiply(A, D)
        
    
    #Hành động tấn công bong bóng dựa trên con tốt nhất trong đàn
    def Attacking(best_sol, sol, dim, a2):
        b = 1
        L = np.multiply((a2 - 1), np.random.uniform(0, 1, dim)) + 1
        D = np.linalg.norm(best_sol - sol)
        return np.multiply(np.multiply(D, np.exp(b * L)), np.cos(2.0 * np.pi * L)) + best_sol
    
    #Cập nhật fitness của mỗi solutions sau mỗi lần lặp
    def update_fitness(self, solutions, fitness_func, fitness, best_sol):
        fitness_list = mWOAPR.calculate_fitness(solutions, fitness_func)
        best_sol_t, fitness_t = mWOAPR.get_best_solution(solutions, fitness_list)
        if fitness_t < fitness:
            return fitness_t, best_sol_t
        
        return fitness, best_sol
    
    def calculate_b(iter, max_iterater):
        return -1 + 2 * (iter / max_iterater)
    def calculate_a1(iter, max_iterater):
        return 2 * (1 - iter / max_iterater)
    def calculate_a2(iter, max_iterater):
        return -1 - (1 / max_iterater * iter)
    def calculate_belta(iter, max_iterater):
        return 1 - iter/ max_iterater
    
    
    def reduce_population(min_population, max_nfes, nfes, population, dim, lb, ub):
        # Đảm bảo nfes không vượt quá max_nfes
        if nfes > max_nfes:
            nfes = max_nfes

        # Tính toán số lượng dân số mới theo công thức đã cho
        new_population = round(((min_population - population) / max_nfes) * nfes + population)
        
        # Đảm bảo new_population không nhỏ hơn min_population
        new_population = max(new_population, min_population)
        
        # Khởi tạo các giải pháp mới dựa trên kích thước dân số mới
        new_solutions = init.initialization(new_population, dim, lb, ub)
        
        return new_solutions # Trả về một quần thể mới
    
    def mWOAPR_handle(self):
        # Lấy dữ liệu hàm test
        fitness_func, ub, lb, dim = get_info(self.function_name)
        # Khởi tạo quần thể
        solutions = init.initialization(self.pop_size, dim, lb, ub)
        # Tạo mảng chứa các giải pháp tốt nhất
        best_solution = []; fitness = np.inf
        best_fitness = []
        
        iter = 0
        while iter < self.max_iterater: 
            #print(f"giá trị tốt nhất hiện tại: {fitness}")
            print(f"Tổng giải pháp còn lại: {len(solutions)}")
            fitness, best_solution = mWOAPR.update_fitness(self, solutions, fitness_func, fitness, best_solution)
            #plot_solutions_update(solutions, best_solution, fitness, lb, ub, iter)
            best_fitness.append(fitness)
            
            a1 = mWOAPR.calculate_a1(iter, self.max_iterater) # a1[2, 0] 
            a2 = mWOAPR.calculate_a2(iter, self.max_iterater) # a2[-1 -2]
            belta = mWOAPR.calculate_belta(iter, self.max_iterater) #belta thay cho 1 [1 0] 

            i = 0
            new_solutions = np.zeros_like(solutions)
            for solution in solutions:
                A = mWOAPR.calculate_A(a1, dim)
                p = np.random.uniform(0, 1)
                if belta < np.linalg.norm(A): # Kham pha
                    if p < 0.5:
                        new_solutions[i, :] = init.initialization(1, dim, lb, ub) # Khởi tạo lại giải pháp đang xét
                    else:
                        new_solutions[i, :] = mWOAPR.Encircling(best_solution, solution, A, dim)
                else: # Khai pha
                    new_solutions[i, :] = mWOAPR.Attacking(best_solution, solution, dim, a2)
                
                i += 1
            solutions = new_solutions
            solutions = mWOAPR.reduce_population(15, 1000, 50, len(solutions), dim, lb, ub)
            
            iter += 1
        plot_results(self.max_iterater, best_fitness)
        print(f"Số giải pháp còn lại: {len(solutions)}")
        return best_solution, fitness