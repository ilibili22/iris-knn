import numpy as np # To calculate the average
from sklearn.datasets import load_iris # iris dataset
from sklearn.model_selection import train_test_split # to split dataset between test and train data
from sklearn.neighbors import KNeighborsClassifier # model for predicting the type of flower
iris = load_iris() # load the iris dataset

# split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(iris['data'],iris['target'], test_size= 0.50, train_size=0.50)  

# create a model with K
model = KNeighborsClassifier(3)

# train the model with training data
model.fit(x_train, y_train)
# print(x_train)
# print(y_train)

# test the model with testing data
testing_model = model.predict(x_test)
# print(x_test)
# print(testing_model)

# use np to calculate the average of right answers
score = np.mean(testing_model == y_test) * 100

# printing the output:
print(f"Test set score: {score:.2f}")