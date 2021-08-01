import pandas as pd
import numpy as np
df = pd.read_csv("F:/Kaggle/Bike sharing Demand/train.csv",
                 parse_dates=['datetime'])
df['year'] = df['datetime'].dt.year
df['month'] = df['datetime'].dt.month
df['day'] = df['datetime'].dt.day
df['hour'] = df['datetime'].dt.hour
df['weekday']=df['datetime'].dt.weekday

#df['season'] = df['season'].astype('category')
#df['weather'] = df['weather'].astype('category')
df_casual = df.drop(columns=['datetime','count', 'registered'])


X = df_casual.drop('casual',axis=1)
y = df_casual['casual']

#######################Grid Search CV#############################

########################### casual ##############################

depth_range = [5,6,7,8,9]
minsplit_range = [10,20,25,30]
minleaf_range = [10,15,20,25]

parameters = dict(max_depth=depth_range,
                  min_samples_split=minsplit_range, 
                  min_samples_leaf=minleaf_range)

from sklearn.model_selection import KFold
kfold = KFold(n_splits=5, random_state=2021,shuffle=True)
from sklearn.model_selection import GridSearchCV
clf = DecisionTreeRegressor(random_state=2021)
cv = GridSearchCV(clf, param_grid=parameters,
                  cv=kfold,scoring='r2')

cv.fit(X,y)

# Best Parameters
print(cv.best_params_)

print(cv.best_score_)

best_casual = cv.best_estimator_

######################### registered ###########################

df_reg = df.drop(columns=['datetime','count', 'casual'])


X = df_reg.drop('registered',axis=1)
y = df_reg['registered']


clf = DecisionTreeRegressor(random_state=2021)
cv = GridSearchCV(clf, param_grid=parameters,
                  cv=kfold,scoring='r2')

cv.fit(X,y)

# Best Parameters
print(cv.best_params_)

print(cv.best_score_)

best_reg = cv.best_estimator_

############# test set pre-processing
test = pd.read_csv("F:/Kaggle/Bike sharing Demand/test.csv",
                 parse_dates=['datetime'])
test['year'] = test['datetime'].dt.year
test['month'] = test['datetime'].dt.month
test['day'] = test['datetime'].dt.day
test['hour'] = test['datetime'].dt.hour
test['weekday']=test['datetime'].dt.weekday


#df['weather'] = df['weather'].astype('category')
test.drop(columns=['datetime'],inplace=True)


pred_casual = best_casual.predict(test) 
pred_reg = best_reg.predict(test)

pred_casual[pred_casual<0] = 0
pred_reg[pred_reg<0] = 0

predictions = np.round(pred_casual + pred_reg)


test = pd.read_csv("F:/Kaggle/Bike sharing Demand/test.csv")
datetime = test['datetime']
submit = pd.DataFrame({'datetime':datetime, 'count':predictions}) 

submit.to_csv("F:/Kaggle/Bike sharing Demand/submit_DT.csv",
              index=False)



