# Test case for reduce_population function
from mWOAPR import mWOAPR


min_population = 10
max_nfes = 500
nfes = 100
population = 20
dim = 5
lb = -10
ub = 10

new_solutions = mWOAPR.reduce_population(min_population, max_nfes, nfes, population, dim, lb, ub)

print("New population size:", len(new_solutions))
print("First solution in new population:", new_solutions)