This project focuses on analyzing the historical stock prices of Google (GOOGL) and predicting future prices using deep learning models. It involves exploratory data analysis (EDA), statistical tests, and the development of a Long Short-Term Memory (LSTM) model for time series prediction.

Project Structure
Data Collection: We use the yfinance library to fetch historical stock data for Google from January 1, 2021, to the present.

Exploratory Data Analysis (EDA):

Visualizing Closing Prices: We plot the closing prices to understand the overall trend.
Descriptive Statistics: Summary statistics are generated to get insights into the data.
Distributions: Histograms for 'Open', 'High', 'Low', 'Close', and 'Volume' are plotted to analyze their distributions.
Moving Averages:

50-Day and 100-Day Moving Averages: We calculate and plot these to smooth out price data and identify trends over different time frames.
Statistical Tests:

T-Test: A T-test is performed to compare the 'High' and 'Low' prices, testing if their means are significantly different.
ANOVA (Analysis of Variance): Used to compare means among 'Open', 'High', 'Low', and 'Close' prices to see if any of these groups have significantly different means.
Correlation Matrix: We compute and visualize the correlation matrix to understand relationships between different price metrics.

Deep Learning Model:

LSTM (Long Short-Term Memory): An advanced type of Recurrent Neural Network (RNN) used for time series prediction. The model is designed to predict future stock prices based on past data.
Model Architecture:
Layers: The model comprises several LSTM layers, each followed by Dropout layers to prevent overfitting. The final layer is a Dense layer to output a single predicted value.
Training: The model is trained on 70% of the data, with the remaining 30% used for testing.
Prediction: The model predicts stock prices, which are then compared to actual prices.
Plotting Results: The predicted prices are plotted against actual prices to visualize the model's performance.

Key Concepts Explained
Time Series
A time series is a sequence of data points recorded at regular intervals over time. In this project, the time series data is Google's stock prices, where each data point represents the stock price at the end of each trading day.

Deep Learning
Deep learning is a subset of machine learning that uses neural networks with multiple layers (hence "deep") to model complex patterns in data. In this project, we use an LSTM model, a type of deep learning model particularly suited for time series forecasting due to its ability to capture temporal dependencies.

Moving Averages
Moving averages are a tool used in time series analysis to smooth out short-term fluctuations and highlight longer-term trends. The 50-day and 100-day moving averages used in this project are commonly employed in stock market analysis.

T-Test
A T-test is a statistical test that compares the means of two groups to see if they are statistically different from each other. In this project, we used it to compare the 'High' and 'Low' prices of Google stock.

ANOVA
ANOVA (Analysis of Variance) is a statistical method used to compare the means of three or more groups. It helps determine if at least one group mean is significantly different from the others. Here, we used it to compare the 'Open', 'High', 'Low', and 'Close' prices.
