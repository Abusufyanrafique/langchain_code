import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

df=pd.read_csv(r"C:\Users\hp\OneDrive\Desktop\Datasets\breast-cancer (1).csv")
print(df)
print(df.shape)

X_train,X_test,y_train,y_test = train_test_split(df.iloc[:,2:],df.iloc[:,1], test_size=0.2)
print(X_train)
print("y_train",y_train)
le = LabelEncoder()
y_train = le.fit_transform(y_train)  
y_test = le.transform(y_test) 

scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
print(X_train)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)
y_pred=knn.predict(X_test)
print(y_pred)
score =accuracy_score(y_pred,y_test)
print("//////////////////////////////////////score/////////////////////////////////////////",score)
scores=[]
for i in range(1,16):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    y_pred=knn.predict(X_test)
    scores.append(accuracy_score(y_test,y_pred))
    print("scores",scores)
    # plt.plot(range(1,16),scores)
    # plt.show()