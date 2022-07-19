import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

start = '2010-01-01'
end = '2019-12-31'

st.title('Stock Trend Predictor')
userInput = st.text_input('Enter stock Ticker', 'AAPL')
df = data.DataReader(userInput, 'yahoo', start, end)

st.subheader('Data 2010-2019')
st.write(df.describe())

st.subheader('Closing Price vs Time chart')
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close, 'b')
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100 MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100, 'r')
plt.plot(df.Close, 'b')
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100ma & 200 MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
plt.plot(df.Close, 'b')
st.pyplot(fig)

dataTraining = pd.DataFrame(df['Close'][:int(len(df) * 0.7)])
dataTesting = pd.DataFrame(df['Close'][int(len(df) * 0.7):])
scaler = MinMaxScaler(feature_range=(0,1))
dataTrainingArray = scaler.fit_transform(dataTraining)


model = load_model("keras_model.h5")

past100Days = dataTraining.tail(100)
finalDF = past100Days.append(dataTesting, ignore_index=True)

inputData = scaler.fit_transform(finalDF)
xTest = []
yTest = []

for i in range(100, inputData.shape[0]):
  xTest.append(inputData[i-100:i])
  yTest.append(inputData[i, 0])
 
xTest, yTest = np.array(xTest), np.array(yTest)
yPred = model.predict(xTest)
scaler = scaler.scale_
 
scaleFactor = 1/scaler[0]
yPred = yPred * scaleFactor
yTest = yTest * scaleFactor

st.subheader('Predicted vs original')
fig2 = plt.figure(figsize=(12,6))
plt.plot(yTest, 'b', label="original Price")
plt.plot(yPred, 'r', label = "Predicted Price")
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)
