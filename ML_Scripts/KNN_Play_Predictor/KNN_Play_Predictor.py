import numpy as np
import pandas as pd
from sklearn import preprocessing 
from sklearn.neighbors import KNeighborsClassifier

def Play_Predictor(data_path):

	#Step 1: Load data

	data = pd.read_csv(data_path)

	print("Size of Actual dataset :", len(data))

	#Step 2: Clean, Prepare and manipulate data 
	feature_names = ['Wether', 'Temperature']

	print("Names of Features\n", feature_names)

	Wether = data.Wether 
	Temperature = data.Temperature 
	play = data.Play

	#creating labelEncoder 
	le = preprocessing.LabelEncoder()

	#Converting string labels into numbers 
	wether_encoded = le.fit_transform(Wether)
	print(wether_encoded)

	#converting string labels into numbers 
	temp_encoded = le.fit_transform(Temperature) 
	label = le.fit_transform(play)

	print(temp_encoded)

	#combinig weather and temp into single list of tuples 
	features = list(zip(wether_encoded,temp_encoded))

	#Step 3: Train Data

	model = KNeighborsClassifier(n_neighbors=3)

	#Train the model using the training sets 
	model.fit(features, label)

	#Step 4: Test data
	print("\nInput for Wether where\n0 : Overcast, 1 : Rainy, 2: Sunny\nEnter Wether :")
	i = int(input())

	print("\nInput for Temperature where\n0 : Cool, 1 : Hot, 2: Mild\nEnter Temperature :")
	j = int(input())
	print("\n")
	
	predicted = model.predict([[i,j]]) # i : Wether, 2 : Temperature

	print("0 : Dont Play\n1 : Play")

	print("prediction is :",predicted)

def main():

	print("---------------Program by Rishikesh Bharat Gawali---------------")

	print("------------------Machine Learning Application------------------")

	print("---------------Play predictor using KNN algorithm---------------")

	print("\n")

	Play_Predictor("Play_Predictor.csv")

if __name__ == "__main__":

	main()