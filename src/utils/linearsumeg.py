import numpy as np
from scipy.optimize import linear_sum_assignment
from colorama import Fore 

cost = np.array([[4,1,3], [2,0,5], [3,2,2], [0,0,0]])

row_ind, col_ind = linear_sum_assignment(cost) 
print(cost) 
print(Fore.LIGHTBLUE_EX + "Row Index: " +str(row_ind) + Fore.RESET)
print(Fore.LIGHTCYAN_EX + "Colulm Index: " +str(col_ind) + Fore.RESET)
print(Fore.LIGHTGREEN_EX + "Actual Values: " + str(cost[row_ind, col_ind]) + Fore.RESET) 
print(Fore.LIGHTMAGENTA_EX + "Total Cost: " + str(cost[row_ind, col_ind].sum()) + Fore.RESET)