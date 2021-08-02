import pandas as pd

df = pd.read_csv("G:/Statistics (Python)/Cases/Real Estate/Housing.csv")
dum_df = pd.get_dummies(df.iloc[:,1:11], drop_first=True)

from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestRegressor

X = dum_df
y = df.iloc[:,0]

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.3, 
                                                    random_state=2018)

model_rf = RandomForestRegressor(random_state=1211)
model_rf.fit( X_train , y_train )
y_pred = model_rf.predict(X_test)


from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
print(mean_squared_error(y_test, y_pred))
print(mean_absolute_error(y_test, y_pred))
print(r2_score(y_test, y_pred))

#################### Grid Search CV ################################
from sklearn.model_selection import GridSearchCV
import numpy as np
parameters = {'max_features': np.arange(1,11)}

from sklearn.model_selection import KFold
kfold = KFold(n_splits=5, random_state=2021,shuffle=True)

model_rf = RandomForestRegressor(random_state=1211)
cv = GridSearchCV(model_rf, param_grid=parameters,
                  cv=kfold,scoring='r2')

cv.fit( X , y )

results_df = pd.DataFrame(cv.cv_results_  )

print(cv.best_params_)

print(cv.best_score_)

print(cv.best_estimator_)

best_model = cv.best_estimator_
############ sorted #############
features = X.columns
importances = best_model.feature_importances_
indices = np.argsort(importances)

plt.title('Feature Importance Plot')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()