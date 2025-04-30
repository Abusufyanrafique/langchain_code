from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score
import numpy as np
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.model_selection import train_test_split


X,y=load_diabetes(return_X_y=True)
print(X.shape)

X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.2, random_state=2)
reg=LinearRegression()
reg.fit(X_train,y_train)
print(reg.coef_)
print(reg.intercept_)

y_pred=reg.predict(X_test)
score =r2_score(y_test,y_pred)
print(score)

class GDRegressor:
    def __init__(self,learning_rate=0.01,epochs=100):
        self.coef_=None
        self.intercept_=0
        self.lr=learning_rate
        self.epochs=epochs
    def fit(self,X_train,y_train):
        self.intercept_=0
        self.coef_ = np.ones(X_train.shape[1])
        for i in range(self.epochs):
            y_hat=np.dot(X_train,self.coef_)+self.intercept_
            print("y_hat shape",y_hat.shape)
            intercept_der=-2*np.mean(y_train - y_hat)
            self.intercept_=self.intercept_ -(self.lr)
            coef_der=-2*np.dot((y_train[i]))
        self.coef_=np.ones(X_train.shape[1])
        print(self.coef_,self.intercept_)
    def predict(self,X_test):
        pass    
gdr=GDRegressor(epochs=10,learning_rate=0.1)
gdr.fit(X_train,y_train)