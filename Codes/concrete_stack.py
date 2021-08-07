import pandas as pd

df = pd.read_csv(r"G:\Statistics (Python)\Cases\Concrete Strength\Concrete_Data.csv")

X = df.iloc[:,:-1]
y = df.iloc[:,-1]

# Import the necessary modules
from sklearn.model_selection import train_test_split 

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, 
                                                    random_state=2021)


from sklearn.linear_model import LinearRegression, Ridge,Lasso
lin_reg = LinearRegression()
ridge = Ridge()
lasso = Lasso()

from sklearn.tree import DecisionTreeRegressor
dtc = DecisionTreeRegressor(random_state=2021)

from sklearn.svm import SVR
svr = SVR()

from sklearn.ensemble import StackingRegressor
models_considered = [('Linear Regression', lin_reg),
                     ('Ridge', ridge),('Lasso',lasso),('Decision Tree',dtc),
                     ('SVR',svr)]

from xgboost import XGBRegressor
clf = XGBRegressor(random_state=2021)

stack = StackingRegressor(estimators = models_considered,
                           final_estimator=clf)

stack.fit(X_train,y_train)

############ w/o passthrough ##################
y_pred = stack.predict(X_test)

from sklearn.metrics import r2_score
print(r2_score(y_test, y_pred))

########### with passthrough ##################
stack = StackingRegressor(estimators = models_considered,
                           final_estimator=clf,
                           passthrough=True)

stack.fit(X_train,y_train)

y_pred = stack.predict(X_test)

print(r2_score(y_test, y_pred))




