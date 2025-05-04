import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.model_selection import train_test_split


df=pd.read_csv(r"C:\Users\hp\OneDrive\Desktop\placement.csv")

print(df)

X = df[['cgpa']]       
y = df['package']            

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_test)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(y_pred)
mae= mean_absolute_error(y_test,y_pred)
mse=mean_squared_error(y_test,y_pred)
rmse=np.sqrt(mean_squared_error(y_test,y_pred))
r2 = r2_score(y_test, y_pred)
print("MAE:", mae)
print("MSE:", mse)
print("RMSE",rmse)
print("R2 Score:", r2)
print(model.coef_)
print(model.intercept_)
sns.scatterplot(x=X_test['cgpa'], y=model.predict(X_test))


plt.show()



