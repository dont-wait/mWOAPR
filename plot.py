import matplotlib.pyplot as plt
import numpy as np
def plot_results(num_iterations, best_fitness):
    iterations = np.arange(1, num_iterations + 1)  # Tạo mảng cho vòng lặp
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, best_fitness, label='Best Fitness', color='blue')
    
    plt.title('Best Fitness Over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Best Fitness Value')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    input("Press Enter to exit...")

# def plot_diversity