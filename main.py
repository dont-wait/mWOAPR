from e_woa import E_WOA
class main:
    def __init__(self, pop_size, function_name, max_iterater):
        self.pop_size = pop_size
        self.function_name = function_name
        self.max_iter = max_iterater
        self.best_sol = None
        self.fitness_sol = None
    def optimize(self):
        self.best_sol, self.fitness_sol = E_WOA.e_woa_handle(self.function_name, self.pop_size, self.max_iter)
    def print_best_solution(self):
        print("Giải pháp tốt nhất:", self.fitness_sol)
            
if __name__ == "__main__":
    optimizer = main(pop_size=30, function_name='F12', max_iterater=100)
    optimizer.optimize()
    optimizer.print_best_solution()        