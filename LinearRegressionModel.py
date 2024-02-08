# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 15:56:24 2023

@author: dell
"""

import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
data= pd.read_csv('C:\\Users\\dell\\Downloads\\archive (1)\\Housing.csv')

#converting yes to 1 and no to 0    
def binary_map(value):
    if value == 'yes':
        return 1
    elif value == 'no':
        return 0
columns_to_convert =['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea' ]
data[columns_to_convert] = data[columns_to_convert].applymap(binary_map)
print(data)

#alloting dummy variable for string data column
dummy_col = pd.get_dummies(data['furnishingstatus'])
dummy_col.head()
dummy_col = pd.get_dummies(data['furnishingstatus'], drop_first=True)
dummy_col.head()
data = pd.concat([data, dummy_col], axis=1)
data.head()
data.drop(['furnishingstatus'], axis=1, inplace=True)
data.head()
from sklearn.linear_model import LinearRegression

x = data.drop(columns=['price'])
y = data['price']
#applying linear regression
model = LinearRegression()
model.fit(x, y)

#residuals
residuals = y - model.predict(x)
#checking homoscedasticity
plt.scatter(model.predict(x), residuals)
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.axhline(0, color='red', linestyle='dashed')
plt.show()

#checking multicollinearity
from statsmodels.stats.outliers_influence import variance_inflation_factor

def calculate_vif(data):
    vif_data = pd.DataFrame()
    vif_data["Variable"] = data.columns
    vif_data["VIF"] = [variance_inflation_factor(data.values, i) for i in range(data.shape[1])]
    return vif_data

vif = calculate_vif(data.drop('price', axis=1))

#splitting the dataset
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)
print(x_train)
print(y_train)
print(x_test)
print(y_test)
#scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
print(x_train)
print(x_test)
#applying linear regression model
from sklearn.linear_model import LinearRegression
classifier = LinearRegression()
classifier.fit(x_train, y_train)

# coefficients = classifier.coef_
# print(coefficients)
# E = classifier.score(x_train,y_train)
# print(E)

# Predicting the Test set results
y_pred = classifier.predict(x_test)
print(y_pred)
ty_pred = classifier.predict(x_train)
print(ty_pred)

#findig mse and r2
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
mape = mean_absolute_percentage_error(y_test, y_pred)
print("MAPE:", mape)
print("Mean Squared Error:", mse)
print("R-squared (R2):", r2)
def mean_absolute_percentage_error(y_true, y_predicted):
    y_true, y_predicted = np.array(y_true), np.array(y_predicted)
    return np.mean(np.abs((y_true - y_predicted) / y_true)) * 100
Tmape = mean_absolute_percentage_error(y_train, ty_pred)
print("TrainMAPE:", Tmape)


#Plotinng a graph for actual vs predicted prices
plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred, color='blue')
plt.title('Actual vs. Predicted Housing Prices')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.show()

#checking if the model is underfitting or overfitting
train_mse = mean_squared_error(y_train, model.predict(x_train))
test_mse = mean_squared_error(y_test, y_pred)


print("Training Mean Squared Error:",train_mse)
print("Testing Mean Squared Error:",test_mse)
print("R-squared:",r2)

from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores = learning_curve(model, x, y, cv=5, scoring='neg_mean_squared_error')
train_rmse = np.sqrt(-train_scores.mean(axis=1))
test_rmse = np.sqrt(-test_scores.mean(axis=1))

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_rmse, label='Training RMSE')
plt.plot(train_sizes, test_rmse, label='Testing RMSE')
plt.xlabel('Training Set Size')
plt.ylabel('RMSE')
plt.legend()
plt.title('Learning Curve')
plt.show()

