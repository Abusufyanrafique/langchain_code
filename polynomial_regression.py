import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


df = pd.read_csv(r"C:\Users\hp\OneDrive\Desktop\polynomial.csv")
print(df.head())  
print(df.columns)

X = df.iloc[:, 1:2].values 
y = df.iloc[:, 2].values    

reg = LinearRegression()
reg.fit(X,y)
predicted_value =reg.predict([[3]])
print(predicted_value)

poly = PolynomialFeatures(degree=2)
x_poly = poly.fit_transform(X)
poly_reg=LinearRegression()
poly_reg.fit(x_poly,y)
result = poly_reg.predict(x_poly)

print("Coefficients:", poly_reg.coef_)
print("Intercept:", poly_reg.intercept_)
print(result)
# sns.scatterplot(x='level', y='salary', data=df)  
# plt.title("level vs salary")
# plt.xlabel("level")
# plt.ylabel("salary")
# plt.grid()
plt.show()
