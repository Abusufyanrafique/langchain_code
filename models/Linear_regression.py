import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


df=pd.read_csv(r"C:\Users\hp\OneDrive\Desktop\placement.csv")

print(df)

X = df[['cgpa']]       
y = df['package']            

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_test)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test.iloc[[0]])
print(y_pred)
print(model.coef_)
print(model.intercept_)
sns.scatterplot(x=X_test['cgpa'], y=model.predict(X_test))


plt.show()



