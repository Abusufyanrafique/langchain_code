from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

X,y=make_regression(n_samples=4,n_features=1,n_informative=1,n_targets=1,noise=80,random_state=13)
lr = LinearRegression()
lr.fit(X,y)
m=78.35063668
b=0
learning_rate=0.01
epochs=1
for i in range(epochs):
    loss_slope=-2*np.sum(y-m*X.ravel()-b)
    b=b-(loss_slope*learning_rate)
# first iteration //////////////////////////////////
# loss_slope=-2*np.sum(y-m*X.ravel()-b)
# step_size=loss_slope*learning_rate
# b=b-step_size
# # second iteration //////////////////////////////////
# loss_slope=-2*np.sum(y-m*X.ravel()-b)
# step_size=loss_slope*learning_rate
# b=b-step_size
# # third iteration //////////////////////////////////
# loss_slope=-2*np.sum(y-m*X.ravel()-b)
# step_size=loss_slope*learning_rate
# b=b-step_size
# # fourth iteration //////////////////////////////////
# loss_slope=-2*np.sum(y-m*X.ravel()-b)
# step_size=loss_slope*learning_rate
# b=b-step_size
# # fourth iteration //////////////////////////////////
# loss_slope=-2*np.sum(y-m*X.ravel()-b)
# step_size=loss_slope*learning_rate
# b=b-step_size
# # fourth iteration //////////////////////////////////
# loss_slope=-2*np.sum(y-m*X.ravel()-b)
# step_size=loss_slope*learning_rate
# b=b-step_size
# # fourth iteration //////////////////////////////////
# loss_slope=-2*np.sum(y-m*X.ravel()-b)
# step_size=loss_slope*learning_rate
# b=b-step_size
print(b)
# print(step_size)
print(loss_slope)
print(lr.coef_)
print(lr.intercept_)
# plt.scatter(X,y)
plt.show()