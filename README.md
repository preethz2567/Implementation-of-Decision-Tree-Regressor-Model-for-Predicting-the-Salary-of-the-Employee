# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. **Import necessary libraries**  
   Use `pandas` for data handling and `LabelEncoder` for encoding categorical variables.

2. **Load the dataset**  
   Read the salary dataset using `pd.read_csv()`.

3. **Check dataset info**  
   Use `.info()` and `.isnull().sum()` to inspect structure and missing values.

4. **Encode categorical features**  
   Apply `LabelEncoder` to convert the `Position` column into numeric form.

5. **Define features and target**  
   - `x`: Feature columns — `Position` and `Level`  
   - `y`: Target column — `Salary`

6. **Split the data**  
   Use `train_test_split()` to split data into training and testing sets (80% train, 20% test).

7. **Initialize Decision Tree Regressor**  
   Create a `DecisionTreeRegressor()` model.

8. **Train the model**  
   Fit the model using training data with `.fit()`.

9. **Make predictions**  
   Use `.predict()` to generate predictions on the test set.

10. **Evaluate the model**  
   Calculate Mean Squared Error (MSE) and R² Score using `metrics`.


## Program:
```

Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: PREETHI D 
RegisterNumber:  212224040250

```
```
import pandas as pd
```
```
df=pd.read_csv("Salary.csv")
df.head()
```
![image](https://github.com/user-attachments/assets/d2a02b9c-501e-4a50-bf83-2088f792574b)
```
df.info()
```
![image](https://github.com/user-attachments/assets/196010ed-b8fe-4860-bbb2-80d9f922643f)

```
df.isnull().sum()
```
![image](https://github.com/user-attachments/assets/7ead255f-870d-4864-b0ab-8284138c62fb)

```
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
```
```
df["Position"]=le.fit_transform(df["Position"])
df.head()
```
![image](https://github.com/user-attachments/assets/1ce55c51-23a5-4941-bf54-cb28286e542d)
```
x = df[["Position", "Level"]]
y = df["Salary"]
```
```
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2, random_state=2)
```
```
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train, y_train)
y_pred=dt.predict(x_test)
```
```
from sklearn import metrics
mse=metrics.mean_squared_error(y_test, y_pred)
mse
```
![image](https://github.com/user-attachments/assets/f50a61f4-dc41-461a-8068-6954c9dcbc9a)
```
r2 = metrics.r2_score(y_test, y_pred)
r2
```
![image](https://github.com/user-attachments/assets/f1597f8f-81d5-45d3-aa37-86e6f61e25e6)
```
dt.predict([[5, 6]])
```
![image](https://github.com/user-attachments/assets/c9b65baf-d1ba-42bc-a89c-dc43e6f33f3a)




## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
