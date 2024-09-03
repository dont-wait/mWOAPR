import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
pause_time = 0.25

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

def plot_solutions_update(solutions, best_solution, fitness, ub, lb, i):
    dim = solutions.shape[1]
    
    if dim == 2:
        if not hasattr(plot_solutions_update, "fig"):
            plot_solutions_update.fig, plot_solutions_update.ax = plt.subplots(figsize=(8, 6))
        
        plot_solutions_update.ax.cla()  # Xóa nội dung cũ
        plot_solutions_update.ax.scatter(solutions[:, 0], solutions[:, 1], c='Blue', label=f'best_sol: {best_solution}')
        plot_solutions_update.ax.scatter(best_solution[0], best_solution[1], c='Red', label=f'fitness: {fitness}', s=100)
        plot_solutions_update.ax.set_title(f'Quần thể vòng lặp thứ {i} (2D)', fontweight="bold")
        plot_solutions_update.ax.set_xlabel('x1')
        plot_solutions_update.ax.set_ylabel('X2')
        plot_solutions_update.ax.set_xlim(lb, ub)
        plot_solutions_update.ax.set_ylim(lb, ub)
        plot_solutions_update.ax.grid()
        plot_solutions_update.ax.legend()
        plt.draw()
        plt.pause(pause_time)
        
    elif dim == 3:
        if not hasattr(plot_solutions_update, "fig"):
            plot_solutions_update.fig = plt.figure(figsize=(8, 6))
            plot_solutions_update.ax = plot_solutions_update.fig.add_subplot(111, projection='3d')
        
        plot_solutions_update.ax.cla()
        plot_solutions_update.ax.scatter(solutions[:, 0], solutions[:, 1], solutions[:, 2], c='Blue', label=f'best_sol: {best_solution}')
        plot_solutions_update.ax.scatter(best_solution[0], best_solution[1], best_solution[2], c='Red', label=f'fitness: {fitness}', s=100)
        plot_solutions_update.ax.set_title(f'Quần thể vòng lặp thứ {i} (3D)', fontweight="bold")
        plot_solutions_update.ax.set_xlabel('X1')
        plot_solutions_update.ax.set_ylabel('X2')
        plot_solutions_update.ax.set_zlabel('X3')
        plot_solutions_update.ax.set_xlim(lb, ub)
        plot_solutions_update.ax.set_ylim(lb, ub)
        plot_solutions_update.ax.set_zlim(lb, ub)
        plot_solutions_update.ax.legend()
        plt.draw()
        plt.pause(pause_time)
        
    else:
        # Giảm chiều xuống 2D bằng PCA
        pca = PCA(n_components=2)
        reduced_solutions = pca.fit_transform(solutions)
        reduced_best_solution = pca.transform(best_solution.reshape(1, -1))

        if not hasattr(plot_solutions_update, "fig"):
            plot_solutions_update.fig, plot_solutions_update.ax = plt.subplots(figsize=(8, 6))
        
        plot_solutions_update.ax.cla()  # Xóa nội dung cũ
        plot_solutions_update.ax.scatter(reduced_solutions[:, 0], reduced_solutions[:, 1], c='Blue', label=f'Quần thể vòng lặp thứ {i} (PCA)')
        plot_solutions_update.ax.scatter(reduced_best_solution[0, 0], reduced_best_solution[0, 1], c='Red', label='Giải pháp tốt nhất', s=100)
        plot_solutions_update.ax.set_title(f'Quần thể vòng lặp thứ {i} (Giảm chiều)', fontweight="bold")
        plot_solutions_update.ax.set_xlabel('PC1')
        plot_solutions_update.ax.set_ylabel('PC2')
        plot_solutions_update.ax.axhline(0, color='black', linewidth=0.5, ls='--')
        plot_solutions_update.ax.axvline(0, color='black', linewidth=0.5, ls='--')
        plot_solutions_update.ax.set_xlim(lb, ub)
        plot_solutions_update.ax.set_ylim(lb, ub)
        plot_solutions_update.ax.grid()
        plot_solutions_update.ax.legend()
        plt.draw()
        plt.pause(pause_time)