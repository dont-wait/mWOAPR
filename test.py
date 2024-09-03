import numpy as np

def divide_half_solutions(solutions, fitness):
    sorted_index = np.argsort(fitness)
    sorted_solutions = solutions[sorted_index]
    
    half_size = len(solutions) // 2  # Sửa lại để lấy nửa kích thước
    half_solutions = sorted_solutions[:half_size]
    
    return half_solutions

# Tạo dữ liệu mẫu
np.random.seed(0)  # Để kết quả có thể tái tạo
solutions = np.random.rand(10, 2)  # 10 giải pháp, mỗi giải pháp có 2 biến
fitness = np.random.rand(10)  # 10 giá trị fitness

# Gọi hàm
half_solutions = divide_half_solutions(solutions, fitness)

# In kết quả
print("Giải pháp ban đầu:")
print(solutions)
print("\nGiá trị fitness:")
print(fitness)
print("\nNửa giải pháp tốt nhất:")
print(half_solutions)

print(len(solutions))