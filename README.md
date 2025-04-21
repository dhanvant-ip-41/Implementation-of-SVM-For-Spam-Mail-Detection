# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import required libraries and read the dataset using pandas.
2. Split the data into train and test data. Fit transform the data using CountVectorizer.
3. Predict Y using Support Vector Classifier(SVC) and calculate the accuracy.
4. Now print the necessary outputs.

## Program:

Program to implement the SVM For Spam Mail Detection.

Developed by: Dhanvant Kumar V                                                                                                             
RegisterNumber: 212224040070

```python
import chardet
file='spam.csv'
with open (file,'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
result

import pandas as pd
data=pd.read_csv("spam.csv",encoding='windows-1252')

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```


## Output:
### ENCODING
![alt text](/image/image.png)
### HEAD()
![alt text](/image/image-1.png)
### INFO()
![alt text](/image/image-2.png)
### SUM OF NULL VALUES
![alt text](/image/image-3.png)
### PREDICTED Y VALUE
![alt text](/image/image-4.png)
### ACCURACY
![alt text](/image/image-5.png)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
