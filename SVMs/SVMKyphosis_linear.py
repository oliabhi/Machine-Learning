import pandas as pd
import numpy as np

df = pd.read_csv("G:/Statistics (Python)/Cases/Kyphosis/Kyphosis.csv")

dum_df = pd.get_dummies(df, drop_first=True)

from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import SVC

X = dum_df.iloc[:,0:3]
y = dum_df.iloc[:,3]

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.3, 
                                                    random_state=42,
                                                    stratify=y)

svc = SVC(probability = True,kernel='linear')
fitSVC = svc.fit(X_train, y_train)
y_pred = fitSVC.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

print(accuracy_score(y_test, y_pred))

# ROC
from sklearn.metrics import roc_curve, roc_auc_score

# Compute predicted probabilities: y_pred_prob
y_pred_prob = svc.predict_proba(X_test)[:,1]

# Generate ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

# Plot ROC curve
import matplotlib.pyplot as plt
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()
roc_auc_score(y_test, y_pred_prob)


########################### Tunning #####################################

########################## Grid Search CV #############################
from sklearn.model_selection import GridSearchCV

C_range = np.array([0.001,0.05,0.01,0.05,0.1,1,1.5,1.7,2,4])
# OR
#C_range = np.linspace(0.001,2)

param_grid = dict( C=C_range)
#cv = StratifiedShuffleSplit(n_splits=5, train_size=2, test_size=None, random_state=42)

from sklearn.model_selection import StratifiedKFold
kfold = StratifiedKFold(n_splits=5, random_state=2020,shuffle=True)
svmGrid = GridSearchCV(SVC(probability=True,kernel='linear'), 
                       param_grid=param_grid, cv=kfold, 
                       scoring="roc_auc")
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
