import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

diabetes = datasets.load_diabetes()
# print(diabetes.keys())
# dict_keys(['data', 'target', 'frame', 'DESCR', 'feature_names', 'data_filename', 'target_filename', 'data_module'])
print(diabetes.DESCR)
# diabetes_x = diabetes.data
# print(diabetes_x)


# diabetes_x_train = diabetes_x[:-30]
# diabetes_x_test = diabetes_x[-30:]

# diabetes_Y_train = diabetes.target[:-30]
# diabetes_Y_test = diabetes.target[-30:]








diabetes_x = np.array([[1],[2],[3]])

print(diabetes_x)


diabetes_x_train = diabetes_x
diabetes_x_test = diabetes_x

diabetes_Y_train = np.array([3,2,4])
diabetes_Y_test = np.array([3,2,4])









model = linear_model.LinearRegression()
model.fit(diabetes_x_train, diabetes_Y_train)

diabetes_Y_predict = model.predict(diabetes_x_test)

print("Mean Square Error : ", mean_squared_error(diabetes_Y_test, diabetes_Y_predict))
print("Weaight : ", model.coef_)
print("intercept : ", model.intercept_)

plt.scatter(diabetes_x_test, diabetes_Y_test)
plt.plot(diabetes_x_test, diabetes_Y_predict)
plt.show()

# Mean Square Error :  3035.0601152912695
# Weaight :  [941.43097333]
# intercept :  153.39713623331698