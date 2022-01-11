#import the lib
import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf
from fbprophet import Prophet
plt.style.use('fivethirtyeight')

st.title('Stock Trend Prediction using LSTM models')
user_input = st.text_input('Enter Stock Ticker', 'ADA-EUR')
#Get the stock quote
df = web.DataReader(user_input, data_source='yahoo', start='2012-01-01', end='2021-11-18')


#Describe Data
st.subheader('Data from 2012 - 2021')
st.write(df.describe())

#Vizualisation

st.subheader('Close stock value')
#Visualize the closing price
fig1 = plt.figure(figsize=(16, 8))
plt.title("Closing price history")
plt.plot(df['Close'])
plt.xlabel("Date", fontsize=18)
plt.ylabel("Close Price USD $", fontsize = 18)
plt.show()
st.pyplot(fig1)


#Create a new df with only the close column
data = df.filter(['Close'])
#convert the dataframe to a numpy array
dataset = data.values
#get the number of rows to train the model on
training_data_len = math.ceil(len(dataset) * .8)

#Scale the data -> almost super advantageous to apply preprocessing normalization before presenting it to a RNN
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

train_data = scaled_data[0:training_data_len, :]
#Split the data into x_train and y_train -> x_train are the independant training variables, y_train are the dependant variables (target var)
x_train = []
y_train = []

for i in range(60, len(train_data)):
  x_train.append(train_data[i-60:i, 0])
  y_train.append(train_data[i, 0])

#Convert the y_train and x_train to numpy arrays
x_train = np.array(x_train)
y_train = np.array(y_train)

#Reshape the data -> in order to have an input (samples, timestamps, feature) for the LSTM model
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_train.shape

#Load the model
model = load_model('LSTM_MODEL.h5')

#Create a new array containing scaled values from index 1752 to 2265 
test_data = scaled_data[training_data_len - 60 : , :]
#Create the datasets x_test and y_test
x_test = [] #values that we want our model to predict
y_test = dataset[training_data_len:, :] # actual values not scaled

for i in range(60, len(test_data)):
  x_test.append(test_data[i-60:i, 0])
  

#Convert the data into numpy array
x_test = np.array(x_test)

#Reshape the data to have it 3 dimensionnal for the LSTM model
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

#Get the model's predicted price values
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions) # -> unscaling the values to get the actual stock prices

#Get the root mean squared error (RMSE) good measure of how good the model predicted the values
# The lows values of RMSE shows that the model is well fit
rmse = np.sqrt(np.mean(predictions - y_test)**2)

#That looks pretty fucking good man
# The model matches almost perfectly the values, it might be naive but we'll see

#Plot the data
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions
#Visualize the data

st.subheader('Prediction plot')
fig2 = plt.figure(figsize=(16, 8))
plt.title('LSTM predictions')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD $', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Validation', 'Predictions'], loc='lower right')
plt.show()
st.pyplot(fig2)
now_date = '2021-11-12'

#Get the quote
apple_quote = web.DataReader(user_input, data_source='yahoo', start='2012-01-01', end=now_date)
#Create a new dataframe
new_df = apple_quote.filter(['Close'])
#get the last 60 day closing price values and convert the df to an array
last_60_days = new_df[-60:].values
#Scale the data to be between 0 and 1
last_60_days_scaled = scaler.transform(last_60_days)
#Create an empty list
X_test = []
#Append the past 60 days to the X_test list
X_test.append(last_60_days_scaled)
#COnvert the X_test to a numpy array
X_test = np.array(X_test)
#Reshape the data to 3 dim -> for the LSTM model
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
#Get the predicted scaled price
pred_price = model.predict(X_test)
#Undo the scaling
pred_price = scaler.inverse_transform(pred_price)
st.subheader(f'Predicted price for {now_date} : {pred_price} USD $')

#Get the quote
apple_quote2 = web.DataReader(user_input, data_source='yahoo', start=now_date, end=now_date)
actual = round(apple_quote2['Close'][0], 2)
st.subheader(f'Actual price of the stock : {actual} USD $')

#############################################################################################################

st.title('Stock Trend Prediction using Prophet model')

data = yf.Ticker(user_input)

hist = data.history(period="max", auto_adjust=True)

st.subheader(f'Historical Data of {user_input} : ')
hist

#Defining the new dataframe

df = pd.DataFrame()
df['ds'] = hist.index
df['y'] = hist['Close'].values


#Defining and fitting the prophet model
m = Prophet(daily_seasonality=False)
m.fit(df)


st.subheader(f'Using Prophet model to make the 1-year dataset prediction for {user_input} : ')
#DIsplay the predicted data in a dataset
future = m.make_future_dataframe(365, freq='D')

forecast = m.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(7)
forecast

st.subheader(f'Plotting the main model components : trend, weekly and yearly')
plot_components = m.plot_components(forecast)
st.pyplot(plot_components)

st.subheader(f'Plotting the predicted trend for {user_input} : ')
plot_forecast = m.plot(forecast)
st.pyplot(plot_forecast)
