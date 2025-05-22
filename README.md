# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
1. Import the required packages.
2. Print the present data and placement data and salary data.
3. Using logistic regression find the predicted values of accuracy confusion matrices.
4. Display the results.

## Program:
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

Developed by: VINOTHINI T

RegisterNumber: 212223040245

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder
data = pd.read_csv('Placement_Data.csv')
data.head()
data1 = data.copy()
data1 = data1.drop(["sl_no", "salary"], axis = 1)
data1.head()
data1.isnull().sum()
data1.duplicated().sum()
le=LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1
x = data1.iloc[:, :-1]
x
y = data1["status"]
y
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = "liblinear")
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
accuracy
from sklearn.metrics import confusion_matrix
confusion = (y_test, y_pred)
confusion
from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test, y_pred)
print(classification_report1)
lr.predict([[1, 80, 1, 90, 1, 1, 90, 1, 0, 85, 1, 85]])
```

## Output:

### Placement Data
![Image](https://github.com/user-attachments/assets/9fd41f0c-79e6-4637-af56-f854d304087f)

### Checking the null() function
![Image](https://github.com/user-attachments/assets/8fb40395-7ddd-4bdf-876a-4ffb1c9bf63c)

### Print Data:
![Image](https://github.com/user-attachments/assets/dbe53128-4842-4900-a28a-7856ba865920)

### Y_prediction array
![Image](https://github.com/user-attachments/assets/71317e6b-09da-47e9-9909-1c975bd32b87)

### Accuracy value
![Image](https://github.com/user-attachments/assets/e49f5ed8-808c-402f-b083-86f59437c2b7)

### Confusion array
![Image](https://github.com/user-attachments/assets/ff3905ab-c461-481f-a903-403c87d76e8c)

### Classification Report
![Image](https://github.com/user-attachments/assets/6769c56f-32d1-4e85-ab4c-de2dd2e6f22e)

### Prediction of LR
![Image](https://github.com/user-attachments/assets/b189fd60-fa27-4b77-aef5-9c4b11fd9c9d)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
