import csv
import numpy as np 
from sklearn.svm import SVR
import matplotlib.pyplot as plt 

dates = []
prices = []
dataset=[]

def getData(filename):
	with open (filename,'r') as csvfile:
		csvFileReader = csv.reader(csvfile)
		next(csvFileReader)
		for row in csvFileReader:
			dates.append(int(row[2].split('-')[0]))
			prices.append(float(row[3]))
	dataset = np.array(prices); return (dataset)
	
x = getData('kotak.csv')
print (x)
