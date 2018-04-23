import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import math
import csv
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM

dataset = pd.read_csv('kotak.csv', usecols=[4])
ClosePrice = dataset


#preprossecing 
ClosePrice = np.reshape(ClosePrice.values, (len(ClosePrice),1)) 
scaler = MinMaxScaler(feature_range=(0, 1))
ClosePrice = scaler.fit_transform(ClosePrice)


# Splitting Training and Testing data
train_Data = int(len(ClosePrice) * 0.75) #75% data as train and 25% as test
test_Data = len(ClosePrice) - train_Data
train_Data, test_Data = ClosePrice[0:train_Data,:],ClosePrice[train_Data:len(ClosePrice),:]


def new_dataset(dataset):
	data_X, data_Y = [], []
	for i in range(len(dataset)-2):
		a = dataset[i:(i+1), 0]
		data_X.append(a)
		data_Y.append(dataset[i+1, 0])
	return np.array(data_X), np.array(data_Y)


trainX, trainY = new_dataset(train_Data)
testX, testY = new_dataset(test_Data)



trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# ANN model
model = Sequential()
model.add(LSTM(32, input_shape=(1, 1), return_sequences = True))
model.add(LSTM(16))
model.add(Dense(1))
model.add(Activation('linear'))


# Fitting model
model.compile(loss='mean_squared_error', optimizer='adagrad',metrics=['mse']) 
history=model.fit(trainX, trainY, epochs=2, batch_size=1, verbose=2)

# Predicting 
trainPredict = model.predict(trainX) #Traning data
testPredict = model.predict(testX) #Testing data


# De-normalizing for ploating
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

# Traning dataset plot
trainPredictPlot = np.empty_like(ClosePrice)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[1:len(trainPredict)+1, :] = trainPredict


# Test dataset plot
testPredictPlot = np.empty_like(ClosePrice)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+3:len(ClosePrice)-1, :] = testPredict


dates = []
prices = []

def getData(filename):
	with open (filename,'r') as csvfile:
		csvFileReader = csv.reader(csvfile)
		next(csvFileReader)
		for row in csvFileReader:
			x=dates.append(int(row[2].split('-')[0]))
			prices.append(float(row[3]))


	return prices,dates

def predict_price_LinearReg(dates, prices):
	dates = np.reshape(dates, (len(dates),1)) 
	prices = np.reshape(prices, (len(prices),1))
	
	linear_mod = linear_model.LinearRegression() # defining the linear regression model
	linear_mod.fit(dates, prices) # fitting the data points in the model
	
	xyz=linear_mod.predict(prices)
	
	return linear_mod.predict(len(prices+1))[0][0],xyz


# Price for next day
last_val = testPredict[-1]
next_val = model.predict(np.reshape(last_val, (1,1,1)))
print ("\n-----Last Day Value:", np.asscalar(last_val))
print ("-----Next Day Value Using ANN:", np.asscalar(last_val+next_val))
if np.asscalar(last_val)<np.asscalar(last_val+next_val):
   print ("\n----Price will go up Buy share----")
else:
   print ("\n----Alert Price will go down----")


prices,dates= (getData('kotak.csv'))
predicted_price,linearP = predict_price_LinearReg(dates, prices)  
print ("\nNext day Using Linear Regression is: ", str(predicted_price))



ClosePrice = scaler.inverse_transform(ClosePrice)
plt.plot(ClosePrice, 'g', label = 'Orignal dataset')
plt.plot(trainPredictPlot, 'r', label = 'Traning set')
plt.plot(testPredictPlot, 'b', label = 'Predicted Price ANN')
plt.plot(linearP, color= 'black', label= 'Linear model') 
plt.legend(loc = 'lower right')
plt.xlabel('Time in Days')
plt.ylabel('Stock prices')
plt.show()


# NMSE is used for evaluating the prediction accuracy of the model
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train NMSE: %.2f' % (trainScore))

# NMSE is used for evaluating the prediction accuracy of the model
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test NMSE: %.2f' % (testScore))

#Plot error graph
plt.plot(history.history['mean_squared_error'])
plt.show()























































































