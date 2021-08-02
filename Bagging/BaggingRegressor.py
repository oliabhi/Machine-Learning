import pandas as pd
df = pd.read_csv("G:/Statistics (Python)/Cases/Real Estate/Housing.csv")
dum_df = pd.get_dummies(df.iloc[:,1:11], drop_first=True)

from sklearn.model_selection import train_test_split 
from sklearn.ensemble import BaggingRegressor

X = dum_df
y = df.iloc[:,0]

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.3, 
                                                    random_state=2018)

# Default: Tree Regressor
model_bg = BaggingRegressor(random_state=2021,oob_score=True,
                            max_features = X_train.shape[1],
                            n_estimators=15,
                            max_samples=X_train.shape[0])

# any other model bagging regressor
from sklearn.linear_model import LinearRegression

lr = LinearRegression()

model_bg = BaggingRegressor(base_estimator = lr,
                            random_state=2021,oob_score=True,
                            max_features = X_train.shape[1],
                            n_estimators=15,
                            max_samples=X_train.shape[0])

#### building the model ######
model_bg.fit( X_train , y_train )

print("Out of Bag Score = " + "{:.4f}".format(model_bg.oob_score_))

y_pred = model_bg.predict(X_test)

from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
print(mean_squared_error(y_test, y_pred))
print(mean_absolute_error(y_test, y_pred))
print(r2_score(y_test, y_pred))

