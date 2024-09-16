# Google Stock Price Analysis and Prediction with LSTM

## Introduction
This project focused on analyzing and predicting the stock prices of Google (GOOGL) using historical data from Yahoo Finance, and using a deep learning model to analyse how well the LTSM model fits onto the original stock price data. The analysis includes exploratory data analysis (EDA), statistical testing, and the use of a Long Short-Term Memory (LSTM) model for time-series forecasting. Additionally, key financial ratios are calculated to provide insight into the company's financial performance.

## Objective
The main goals of this project are:
1. Analyze the historical stock data of Google using EDA techniques.
2. Compute fundamental financial ratios such as P/E ratio, P/B ratio, and ROE.
3. Perform statistical tests, including T-tests and ANOVA, on stock price data.
4. Build and train an LSTM model to predict future stock prices.
5. Visualize the performance of the LSTM model by comparing actual vs. predicted prices.

## Requirements
To run this project, you will need the following Python libraries:
- `numpy`: Numerical operations
- `pandas`: Data manipulation
- `matplotlib`: Data visualization
- `yfinance`: Fetching stock data
- `plotly.express`: Interactive visualizations
- `scikit-learn`: Data scaling and statistical tests
- `tensorflow.keras`: Deep learning (LSTM model)

Data Acquisition
The data is fetched using the yfinance library for the stock ticker GOOGL (Google). Historical stock data is retrieved from January 1, 2021, up to the current date. The dataset includes the following columns:

Open price
High price
Low price
Close price
Volume of shares traded

### Exploratory Data Analysis (EDA)
Several analyses and visualizations are performed:

50-day and 100-day moving averages are fitted over the closing prices:
![wow](https://github.com/user-attachments/assets/52f626b5-12c2-45ae-b30a-61d204e60e8e)

Correlation Matrix: A heatmap to show the correlation between different stock price metrics:
![correlation_matrix](https://github.com/user-attachments/assets/a28cbc81-c742-4200-a5d5-5f53a9623d70)

Statistical Analysis
Two statistical tests are performed:

T-test: A t-test is used to compare the means of the High and Low prices.
ANOVA: An ANOVA test is conducted to compare the means of the Open, High, Low, and Close prices.
The results of these tests help evaluate whether the differences in stock price metrics are statistically significant.

LSTM Model for Stock Price Prediction
The project builds an LSTM model to predict future stock prices based on historical data:

Prediction: The model predicts future prices based on the test data, and results are scaled back to the original values.
Visualization: A plot comparing the actual and predicted stock prices is generated:
![prediction](https://github.com/user-attachments/assets/35f4cec7-93c6-4150-94f4-304e512ea15c)

