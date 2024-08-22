# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph.
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Infancia Felcy P
RegisterNumber: 212223040067

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse) 
*/
```

## Output:

### Data Set:
![image](https://github.com/user-attachments/assets/af30833d-3326-48ca-a0c7-53c114cbcf8b)
### Head Value:
![image](https://github.com/user-attachments/assets/f6f29d4e-9d83-4036-b34f-f52050e411dc)
### Tail Value:
![image](https://github.com/user-attachments/assets/490da0df-d001-4c93-9d69-26c5882a5494)
### X and Y values:
![image](https://github.com/user-attachments/assets/d4e09562-f29c-4724-b225-e4114b1e92d2)
### Predicition Values:
![image](https://github.com/user-attachments/assets/99f21adc-dc06-430c-aa44-ec01e4f774ac)
### MSE MAE And RSM:
![image](https://github.com/user-attachments/assets/58def237-6fec-4e18-a123-6448caa6d062)
### Training Set:
![image](https://github.com/user-attachments/assets/fa675edc-8aa0-4551-8b6e-05ecc1f2f9ff)
### Testing Set:
![image](https://github.com/user-attachments/assets/a239c1a5-2192-419b-a273-dd1af37a7df1)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
