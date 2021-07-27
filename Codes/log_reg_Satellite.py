import pandas as pd
import numpy as np

df = pd.read_csv(r"G:\Statistics (Python)\Cases\Satellite Imaging\Satellite.csv",
                 sep=";")

from sklearn.preprocessing import LabelEncoder
lbcode = LabelEncoder()

X = df.iloc[:,:-1]
y = df.iloc[:,-1]

y = lbcode.fit_transform(y)

# Import the necessary modules
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score


# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, 
                                                    random_state=2021,
                                                    stratify=y)

# Create the classifier: logreg
logreg = LogisticRegression()
# Fit the classifier to the training data
logreg.fit(X_train,y_train)
# Predict the labels of the test set: y_pred
y_pred = logreg.predict(X_test)

# Coeff
print(logreg.coef_)
print(logreg.intercept_)

# Compute and print the confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test,y_pred))

################ Multiclass ROC #############################

# Import necessary modules
from sklearn.metrics import roc_auc_score

# Compute predicted probabilities: y_pred_prob
y_pred_prob = logreg.predict_proba(X_test)

roc_auc_score(y_test, y_pred_prob,multi_class='ovr',
              average='macro')

#########################K-Fold CV####################################
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
logreg = LogisticRegression()
kfold = StratifiedKFold(n_splits=5, random_state=2021,
                        shuffle=True)
results = cross_val_score(logreg, X, y, cv=kfold, 
                          scoring='roc_auc_ovr')
AUCs = results
print(AUCs)
print("Mean AUC: %.2f" % (AUCs.mean()))

##########################################################