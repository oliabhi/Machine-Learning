import pandas as pd
import numpy as np

df = pd.read_csv("G:/Statistics (Python)/Cases/Sonar/Sonar.csv")

dum_df = pd.get_dummies(df, drop_first=True)

from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import SVC

X = dum_df.iloc[:,:-1]
y = dum_df.iloc[:,-1]


########################### Tunning Linear #####################################

########################## Grid Search CV #############################
from sklearn.model_selection import GridSearchCV

C_range = np.linspace(0.001,2)

param_grid = dict( C=C_range)
#cv = StratifiedShuffleSplit(n_splits=5, train_size=2, test_size=None, random_state=42)

from sklearn.model_selection import StratifiedKFold
kfold = StratifiedKFold(n_splits=5, random_state=2020,shuffle=True)
svmGrid = GridSearchCV(SVC(probability=True,kernel='linear'), 
                       param_grid=param_grid, cv=kfold, 
                       scoring="roc_auc",verbose=3)
svmGrid.fit(X, y)

# Best Parameters
print(svmGrid.best_params_)
#print(svmGrid.cv_results_)
print(svmGrid.best_score_)

########################## Random Grid Search CV #################### 
from sklearn.model_selection import RandomizedSearchCV

C_range = np.linspace(0.001,4,100)
param_grid = dict( C=C_range)

rcv = RandomizedSearchCV(SVC(probability=True,kernel='linear'),
                        random_state=2021,
                        param_distributions=param_grid ,
                        cv=kfold,scoring='roc_auc',n_iter=10)

rcv.fit( X , y)

print(rcv.best_params_)

print(rcv.best_score_)  

print(rcv.best_estimator_)

############################## Tunning for RBF ###############@@#########
from sklearn.model_selection import GridSearchCV

C_range = np.linspace(0.01,4,10)

gamma_range = np.linspace(0.001,10,10)

parameters = dict(gamma=gamma_range, C=C_range)

svc = SVC(probability=True,kernel='rbf')

from sklearn.model_selection import StratifiedKFold
kfold = StratifiedKFold(n_splits=5, random_state=2021,shuffle=True)
svmGrid = GridSearchCV(svc, param_grid=parameters, cv=kfold,
                       scoring='roc_auc',verbose=3)
svmGrid.fit(X, y)

# Best Parameters
print(svmGrid.best_params_)

print(svmGrid.best_score_)

# Table of Grid Search CV Results
df_cv = pd.DataFrame(svmGrid.cv_results_  )


########################## Random Grid Search CV #################### 
from sklearn.model_selection import RandomizedSearchCV

C_range = np.linspace(0.01,5,100)
gamma_range = np.linspace(0.001,15,100)

rcv = RandomizedSearchCV(SVC(probability=True,kernel='rbf'),
                        random_state=2021,param_distributions=parameters ,
                        cv=kfold,scoring='roc_auc',n_iter=20)

rcv.fit( X , y)

print(rcv.best_params_)

print(rcv.best_score_)  

print(rcv.best_estimator_)

best_model = rcv.best_estimator_

best_model = SVC(probability=True,kernel='rbf',
                 gamma = 1.1119999999999999,
                 C = 3.5566666666666666)