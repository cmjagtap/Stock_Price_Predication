import csv
import numpy as np 
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.layers import Dense
from keras.models import Sequential

dates = []
prices = []

def getData(filename):
	with open (filename,'r') as csvfile:
		csvFileReader = csv.reader(csvfile)
		next(csvFileReader)
		for row in csvFileReader:
			x=dates.append(int(row[2].split('-')[0]))
			prices.append(float(row[3]))


	return prices


# Create dataset matrix (X=t and Y=t+1)
def create_dataset(dataset):
       dataX = [dataset[n+1] for n in range(len(dataset)-2)]
       return np.array(dataX), dataset[2:]

def stock_prediction(dataset):
        
    trainX, trainY = create_dataset(dataset)

    # Create and fit Multilinear Perceptron model
    model = Sequential()
    model.add(Dense(8, input_dim=1, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, nb_epoch=50, batch_size=2, verbose=2)

    # Our prediction for tomorrow
    prediction = model.predict(np.array([dataset[0]]))
    result = '---The price will move from %s to %s---' % (dataset[0], prediction[0][0])
    if dataset[0]<prediction[0][0]:
    	print("---------Price will Go up Buy Stock--------")
    else:
    	print("---------Alert The price will go down------- ")
    return result
print (stock_prediction(getData('kotak.csv')))
