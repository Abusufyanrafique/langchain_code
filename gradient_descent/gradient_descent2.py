import numpy as np


X = np.array([1, 2, 3])   
y = np.array([2, 4, 6])   

m = 1                   
b = 0                   
learning_rate = 0.1    
y_pred = m * X + b       # ŷ = m * X + b

errors = y - y_pred      # errors = y - ŷ

sum_errors = np.sum(errors)

loss_slope = -2 * sum_errors

# Step 5: Calculate step size
step_size = loss_slope * learning_rate

# Step 6: Update the intercept b
b = b - step_size
print("Updated value of b:", b)
