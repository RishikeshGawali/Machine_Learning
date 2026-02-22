from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics

def RishikeshSVM():

    # Load datasets
    cancer = datasets.load_breast_cancer()

    # Print the name of the 13 features
    print("Feature of the cancer dataset:", cancer.feature_names)

    # Print the lable type of cancer('malignant','begin')
    print("Lable of the cancer datset:",cancer.target_names)

    # Print data(feature)shape
    print("Shape of dataset is:",cancer.data.shape)

    # Print the cancer data feature (top 5 records)
    print("Frist 5 records are:")
    print(cancer.data[0:5])

    # Print the cancer lable (0:malignant,1:benign)
    print("Traget of dataset :",cancer.target)

    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size = 0.3, random_state = 109) #70% training and 30% test

    #Create a svm Classifier
    clf = svm.SVC(kernel = 'linear') # Linear Karenl

    # Train the model using the training sets
    clf.fit(X_train, y_train)

    # Predict the responce for test dataset
    y_pred = clf.predict(X_test)

    # Model Accuracy : how often is the classifier correst?
    print("Accuracy of the model is:",metrics.accuracy_score(y_test, y_pred)*100)
    

def main():
    print("---------- Rishikesh Bharat Gawali ----------")
    print("---------- Breast Cancer Dataset ----------")
    print("---------- Support Vector Machine ----------")

    RishikeshSVM()
    
if __name__ == "__main__":
    main()