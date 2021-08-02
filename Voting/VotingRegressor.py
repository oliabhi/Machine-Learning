import pandas as pd
import numpy as np

Housing = pd.read_csv("F:/Python Material/ML with Python/Cases/Real Estate/Housing.csv")
dum_Housing = pd.get_dummies(Housing.iloc[:,1:11], 
                             drop_first=True)

from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

from sklearn.ensemble import VotingRegressor
X = dum_Housing
y = Housing.iloc[:,0]

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, 
                                                    random_state=42)

dtr = DecisionTreeRegressor(max_depth=4)
lr = LinearRegression()
svr = SVR()

# Average
Voting = VotingRegressor(estimators=[('DT',dtr),
                                     ('LR',lr),('SV',svr)])

#OR Weighted Average
Voting = VotingRegressor(estimators=[('DT',dtr),
                                     ('LR',lr),('SV',svr)],
                                     weights=np.array([0.2,0.5,
                                                       0.3]))

Voting.fit(X_train,y_train)
y_pred = Voting.predict(X_test)

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error,r2_score
print(mean_squared_error(y_test, y_pred) ** 0.5)
print(mean_absolute_error(y_test, y_pred))
print(r2_score(y_test, y_pred))

