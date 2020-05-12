#Multiple Linear Regression
 
# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X), dtype=np.float)


#Avoiding Dummy Variavble Trap
X = X[:, 1:]

# Splitting the dataset into Training set and Testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)    

"""# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

#Fitting Multiple Linear Regression to Test Set
from sklearn.linear_model import LinearRegression 
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Predicting Test Set Results
y_pred = regressor.predict(X_test)

# #Building an optimal model for backward propagation
# import statsmodels.formula.api as sm
# import statsmodels.regression.linear_model as lm
# X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1)
# X_opt = X[:, [0, 1, 2, 3, 4, 5]]
# regressor_OLS = lm.OLS(endog = y, exog = X_opt).fit()
# regressor_OLS.summary()

# X_opt = X[:, [0, 1, 3, 4, 5]]
# regressor_OLS = lm.OLS(endog = y, exog = X_opt).fit()
# regressor_OLS.summary()

# X_opt = X[:, [0, 3, 5]]
# regressor_OLS = lm.OLS(endog = y, exog = X_opt).fit()
# regressor_OLS.summary()

# X_opt = X[:, [0, 3]]
# regressor_OLS = lm.OLS(endog = y, exog = X_opt).fit()
# regressor_OLS.summary()



