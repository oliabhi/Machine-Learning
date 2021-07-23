import pandas as pd

telecom = pd.read_csv("F:/Python Material/ML with Python/Datasets/Telecom.csv")

dum_telecom = pd.get_dummies(telecom, drop_first=True)

from sklearn.model_selection import StratifiedKFold
#from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

from sklearn.naive_bayes import MultinomialNB

X = dum_telecom.iloc[:,0:2]
y = dum_telecom.iloc[:,2]

#kfold = KFold(n_splits=5, random_state=42)
kfold = StratifiedKFold(n_splits=5, random_state=2021,
                        shuffle=True)
multinomial = MultinomialNB()
# cross_val_score()
## 1. Fits the model for every fold combination
## 2. Evaluates the model fitted for every fold combination
results = cross_val_score(multinomial, X, y, cv=kfold, 
                          scoring='roc_auc')
print(results)
print("ROC AUC: %.4f (%.4f)" % (results.mean(), results.std()))

