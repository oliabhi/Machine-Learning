import pandas as pd

df = pd.read_csv("G:/Statistics (Python)/Cases/Real Estate/Housing.csv")
dum_df = pd.get_dummies(df, drop_first=True)

X = dum_df.iloc[:,1:11]
y = df.iloc[:,0]

from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
import numpy as np

parameters = {'n_neighbors': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]}
# OR
parameters = {'n_neighbors': np.arange(1,16)}
print(parameters)

knn = KNeighborsRegressor()

from sklearn.model_selection import KFold
kfold = KFold(n_splits=5 , random_state=42,shuffle=True)

cv = GridSearchCV(knn, param_grid=parameters,
                  cv=kfold,scoring='r2',verbose=3)
cv.fit( X , y )
#pd.DataFrame(cv.cv_results_  )
print(cv.best_params_)
print(cv.best_score_)

print(cv.best_estimator_)

############################# Randomized Search ########################
from sklearn.model_selection import RandomizedSearchCV

parameters = {'n_neighbors': np.arange(1,101)}
print(parameters)

knn = KNeighborsRegressor()

from sklearn.model_selection import KFold
kfold = KFold(n_splits=5 , random_state=42,shuffle=True)

rcv = RandomizedSearchCV(knn, param_distributions=parameters,
                  cv=kfold,scoring='r2',n_iter=20,
                  random_state = 2020,verbose=3)
rcv.fit( X , y )

print(rcv.best_params_)
print(rcv.best_score_)
df_rnd = pd.DataFrame(rcv.cv_results_)
print(rcv.best_estimator_)

