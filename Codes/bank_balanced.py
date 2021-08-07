import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

df = pd.read_csv("G:/Statistics (Python)/Cases/bank/bank.csv",sep=";")

dum_df = pd.get_dummies(df, drop_first=True)

X = dum_df.iloc[:,:-1]
y = dum_df.iloc[:,-1]

# Import the necessary modules
from sklearn.svm import SVC

svc = SVC(probability=True,kernel='rbf')

svc.fit(X, y)

results = cross_val_score(svc, X, y, cv=kfold, 
                          scoring='roc_auc')
print(results)
print("ROC AUC: %.4f (%.4f)" % (results.mean(), results.std()))

################### Over-Sampling(Naive) ###############

from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=2021)
X_resampled, y_resampled = ros.fit_resample(X, y)

results = cross_val_score(svc, X_resampled, y_resampled, cv=kfold, 
                          scoring='roc_auc')
print(results)
print("ROC AUC: %.4f (%.4f)" % (results.mean(), results.std()))


################# Over-Sampling(SMOTE) #################

from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=2021)
X_resampled, y_resampled = smote.fit_resample(X, y)

results = cross_val_score(svc, X_resampled, y_resampled, cv=kfold, 
                          scoring='roc_auc')
print(results)
print("ROC AUC: %.4f (%.4f)" % (results.mean(), results.std()))


################# Over-Sampling(ADASYN) #################

from imblearn.over_sampling import ADASYN
adasyn = ADASYN(random_state=2021)
X_resampled, y_resampled = adasyn.fit_resample(X, y)

results = cross_val_score(svc, X_resampled, y_resampled, cv=kfold, 
                          scoring='roc_auc')
print(results)
print("ROC AUC: %.4f (%.4f)" % (results.mean(), results.std()))



