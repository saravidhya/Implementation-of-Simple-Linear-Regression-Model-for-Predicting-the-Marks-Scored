# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

STEP 1. Start the program.

STEP 2.Import the standard Libraries.

STEP 3.Set variables for assigning dataset values.

STEP 4.Import linear regression from sklearn.

STEP 5.Assign the points for representing in the graph.

STEP 6.Predict the regression for marks by using the representation of the graph.

STEP 7.End the program.


## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Vidhiya Lakshmi S
RegisterNumber: 212223230238

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv("student_scores.csv")
df.head()
df.tail()
X=df.iloc[:,:-1].values
X
Y=df.iloc[:,1].values
Y
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
Y_pred
Y_test
plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours Vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(X_test,Y_test,color='purple')
plt.plot(X_train,regressor.predict(X_train),color='yellow')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE = ",rmse)
*/
```

## Output:

![Screenshot 2024-08-20 215802](https://github.com/user-attachments/assets/fa338804-ebea-4ddd-8468-54d307371ce3)

![image](https://github.com/user-attachments/assets/3c16f8a2-87bc-4626-bcb7-4647b549fdf5)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
