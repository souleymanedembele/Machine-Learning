# import packages
import numpy as np
from sklearn import linear_model
import sklearn.metrics as sm
import matplotlib.pyplot as plt

input_file = 'data_linear_regression.txt'

data = np.loadtxt(input_file, delimiter=',')

#Slice the data
X, y = data[:, :-1], data[:, -1]

#Split data into training and testing set
num_training = int(0.8*len(X))
num_test = len(X) - num_training

#Split the target into train/test set
X_train, y_train = X[:num_training], y[:num_training]

X_test, y_test = X[num_training:], y[num_training:]

#Create linear regression object
regressor = linear_model.LinearRegression()

# train the model
regressor.fit(X_train, y_train)

#make prediction
y_test_predict = regressor.predict(X_test)
# Plot output
plt.scatter(X_test, y_test, color='green')
plt.plot(X_test, y_test_predict, color='black', linewidth=4)
plt.xticks(())
plt.yticks(())
plt.show()

print("Performance:")
print("Mean absolute error=", round(sm.mean_absolute_error(y_test, y_test_predict), 2))
print("Mean square error=", round(sm.mean_squared_error(y_test, y_test_predict), 2))
print("Median absolute error=", round(sm.median_absolute_error(y_test, y_test_predict), 2))
print("Explain variance score=", round(sm.explained_variance_score(y_test, y_test_predict), 2))
print("R2 score=", round(sm.r2_score(y_test, y_test_predict), 2))
