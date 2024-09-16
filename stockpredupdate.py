# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 18:22:30 2024

@author: mike_
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
from scipy.stats import ttest_ind, f_oneway

# CREATE TICKER INSTANCE FOR google
google = yf.Ticker("GOOGL")

# GET TODAY'S DATE AND CONVERT IT TO A STRING WITH YYYY-MM-DD FORMAT (YFINANCE EXPECTS THAT FORMAT)
end_date = datetime.now().strftime('%Y-%m-%d')
google_hist = google.history(start='2021-01-01', end=end_date)

# Select columns for plotting
google_main = google_hist.drop(['Dividends', 'Stock Splits'], axis=1)

# Plot Closing Prices
plt.figure(figsize=(11, 6))
plt.title('Stock Closing Prices', fontsize=15)
plt.xlabel('Trading Days', fontsize=11)
plt.ylabel('Closing Price', fontsize=11)
plt.plot(google_main.Close)
plt.show()

# Display statistics
print(google_hist.describe())

# Plotting histograms showing distribution of the features from google.hist data
fig, axs = plt.subplots(3, 2, figsize=(15, 12), tight_layout=True)

# Close histogram
axs[0, 0].hist(google_hist['Close'].dropna(), bins=50, color='blue', alpha=0.7)
axs[0, 0].set_title('Distribution of Close Prices')

# Volume histogram
axs[0, 1].hist(google_hist['Volume'].dropna(), bins=50, color='green', alpha=0.7)
axs[0, 1].set_title('Distribution of Volume')

# Open histogram
axs[1, 0].hist(google_hist['Open'].dropna(), bins=50, color='purple', alpha=0.7)
axs[1, 0].set_title('Distribution of Open Prices')

# High histogram
axs[1, 1].hist(google_hist['High'].dropna(), bins=50, color='red', alpha=0.7)
axs[1, 1].set_title('Distribution of High Prices')

# Low histogram
axs[2, 0].hist(google_hist['Low'].dropna(), bins=50, color='orange', alpha=0.7)
axs[2, 0].set_title('Distribution of Low Prices')


plt.show()

# Finding 50 and 100 days moving averages
moving_avg50 = google_main.Close.rolling(50).mean()
moving_avg100 = google_main.Close.rolling(100).mean()

plt.figure(figsize=(11, 6))
plt.title('50 & 100 Days Moving Averages fits of Closing Prices', fontsize=15)
plt.xlabel('Trading Days', fontsize=11)
plt.ylabel('Closing Price', fontsize=11)
plt.plot(google_main.Close, label='Close')
plt.plot(moving_avg50, 'red', label='50-Day MA')
plt.plot(moving_avg100, 'green', label='100-Day MA')
plt.savefig('wow.png', dpi=300)
plt.legend()
plt.show()

# Finding correlation between variables
correlation = google_hist.corr()
fig = px.imshow(correlation, text_auto=True, title='Correlation Matrix')
fig.update_layout(title_text='Correlation Matrix', title_x=0.5, template='plotly_dark')
fig.show()
fig.write_image('correlation_matrix.png')

# Statistical Tests
# T-test comparing 'High' and 'Low' prices
t_stat, p_value = ttest_ind(google_hist['High'], google_hist['Low'])

print('t-test statistic:', t_stat, 'p-value:', p_value)
# ANOVA used for comparing the means between more than one variables:
anova_stat, anova_p_value = f_oneway(google_hist['Open'], google_hist['High'], google_hist['Low'], google_hist['Close'])

print('ANOVA-test statistic:', anova_stat, 'p-value:', anova_p_value)

# Splitting the dataset into training and testing data
data_training = pd.DataFrame(google_main['Close'][0: int(len(google_main) * 0.7)])
data_testing = pd.DataFrame(google_main['Close'][int(len(google_main) * 0.7): int(len(google_main))])

# Normalizing the data
scaler = MinMaxScaler(feature_range=(0, 1))
data_training_array = scaler.fit_transform(data_training)

x_train = []
y_train = []

for i in range(100, data_training_array.shape[0]):
    x_train.append(data_training_array[i - 100:i])
    y_train.append(data_training_array[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

# Deep Learning
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential

# Building LSTM model
model = Sequential()
model.add(LSTM(units=50, activation= 'relu' , return_sequences= True, input_shape= (x_train.shape[1],1) ))
model.add(Dropout(0.2))

model.add(LSTM(units=60, activation= 'relu' , return_sequences= True))
model.add(Dropout(0.3))

model.add(LSTM(units=80, activation= 'relu' , return_sequences= True))
model.add(Dropout(0.4))

model.add(LSTM(units=120, activation= 'relu'))
model.add(Dropout(0.5))
          
model.add(Dense(units=1))

model.compile(optimizer='adam' , loss='mean_squared_error')

# Training model
history = model.fit( x_train, y_train, epochs=50)
model.save('keras_stock_prediction_model.keras')

prev_100_days = data_training.tail(100)
new_df = pd.concat([prev_100_days,data_testing], ignore_index=True)
new_df.head()
input_data = scaler.fit_transform(new_df)

x_test = []
y_test = []


for i in range(100 , input_data.shape[0]):
    x_test.append(input_data[i-100 : i])
    y_test.append(input_data[i, 0])

# converting to numpy arrays
x_test, y_test = np.array(x_test), np.array(y_test)

# making predictions (y-hat)

y_predicted = model.predict(x_test)
y_predicted.shape

# scale up the values
print(scaler.scale_)  # returns the scalability factor

scale  = 1 / scaler.scale_[0]
y_test = y_test * scale 
y_predicted = y_predicted * scale 

# Plot of Original vs Predicted graph
plt.figure(figsize=(11,6))
plt.title('Original vs Predicted Price Graph')
plt.plot(y_test, 'b', label='Original price')
plt.plot(y_predicted, 'r', label='Predicted price')
plt.xlabel('Time' , fontsize = 10)
plt.ylabel('Price', fontsize = 10)

# (100 days of training data)
plt.axvline(x=100, color='g', linestyle='--', label='Start of Testing Data')

# Annotating the graph to show each section
plt.text(20, y_test.max(), 'Last 100 Days of Training Data', fontsize=10, color='green', verticalalignment='center')
plt.text(120, y_test.max(), 'Testing Data', fontsize=10, color='green', verticalalignment='center')

plt.legend()
plt.savefig('prediction',dpi=300)
plt.show()

# Adding Financial Metrics
def calculate_financial_ratios(ticker):
    info = ticker.info
    pe_ratio = info.get('forwardEps', None) and info.get('forwardEps', None) / info.get('currentPrice', None)
    pb_ratio = info.get('priceToBook', None)
    roe = info.get('returnOnEquity', None)
    
    ratios = {
        'P/E Ratio': pe_ratio,
        'P/B Ratio': pb_ratio,
        'ROE': roe
    }
    return ratios

ratios = calculate_financial_ratios(google)
print("Financial Ratios:\n", ratios)
