# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 06:47:32 2023

@author: dell
"""

import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

tickers= ['RELIANCE.NS']
stock_data= yf.download(tickers, start='2022-07-21', end='2023-11-21')

#%%
#x= list(range(len(df)))
stock_data['Trend'] = range(1, len(stock_data) + 1)

# Selecting the closing prices
stock_data['Close'] = stock_data[['Close']]
#%%

# Adding columns for the features (e.g., past prices) and the target variable
for i in range(1, len(stock_data)):
    stock_data[f'Past_{i}_Days'] = stock_data['Close'].shift(i)
#%%
# Adding a column for the target variable (next day's closing price)
stock_data['Target'] = stock_data['Close'].shift(-1)

# Drop rows with NaN values created by shifting
stock_data = stock_data.dropna()

#%%
X = stock_data[['Target', 'Trend']]
y = stock_data['Close']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred= model.predict(X_test)

from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R-squared (R2):", r2)


plt.scatter(X_test['Trend'], y_test, color='black', label='Actual Prices')
plt.plot(X_test['Trend'], y_pred, color='blue', linewidth=3, label='Predicted Prices')
plt.xlabel('Trend')
plt.ylabel('Closing Price')
plt.legend()
plt.show()

# Use the most recent data for prediction
#latest_data = X.iloc[-1:]

# Predict the next ten days
#future_dates = pd.date_range(start=stock_data.index[-1], periods=11, freq='B')[1:]
#future_predictions = pd.DataFrame(index=future_dates, columns=['Predicted Price'])

#for i, date in enumerate(future_dates):
    #if i == 0:
        #future_predictions.loc[date] = model.predict(latest_data)[0]
    #else:
        #latest_data = pd.DataFrame(index=[date], columns=['Close'], data=future_predictions.iloc[i-1:i].values)
        #future_predictions.loc[date] = model.predict(latest_data)[0]

#print(future_predictions)

