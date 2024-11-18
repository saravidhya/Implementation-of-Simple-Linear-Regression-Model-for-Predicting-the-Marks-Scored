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
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error

df=pd.read_csv('/student_scores.csv')
print(df.head())
print(df.tail())
```
## OUTPUT:

![image](https://github.com/user-attachments/assets/6935bc03-7335-4fc7-9f6b-c4f0aab54181)

```
X=df.iloc[:,:-1].values
print(X)
Y=df.iloc[:,1].values
print(Y)
````

## Output:

![image](https://github.com/user-attachments/assets/158653ec-d5c0-4023-a0a8-2b4c7cbc458e)
```
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)
X_train.shape
```
## OUTPUT

![image](https://github.com/user-attachments/assets/abd5cf21-91f7-4df0-a757-5d5acc5dc1f4)

```
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,Y_train)
X
Y_pred=reg.predict(X_test)
print(Y_pred)
print(Y_test)
```
## OUTPUT

![image](https://github.com/user-attachments/assets/76eeca9f-534c-43cf-9964-ba284a0e0bc6)

```
plt.scatter(X_train,Y_train,color="green")
plt.plot(X_train,reg.predict(X_train),color="red")
plt.title('Trainig set(H vs S)')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(X_test,Y_test,color="blue")
plt.plot(X_train,reg.predict(X_train),color="silver")
plt.title('Test set(H vs S)')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```
## Output

![image](https://github.com/user-attachments/assets/763dcc71-0302-4ab8-bff1-316c11d7a3f3)
![image](https://github.com/user-attachments/assets/76bebaa9-08e0-4ceb-a306-5e84575684de)

```
mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE = ",rmse)
```
## OUTPUT

![image](https://github.com/user-attachments/assets/d783abd5-c735-4fb8-807d-e1d93182fb74)


## Result:

Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
