import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score


df = pd.read_csv(r"G:\Statistics (Python)\Cases\Sonar\Sonar.csv")

dum_df = pd.get_dummies(df,drop_first=True)

X = dum_df.drop('Class_R',axis=1)
y = dum_df['Class_R']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, 
                                                    random_state=2021,
                                                    stratify=y)

logreg = LogisticRegression()

logreg.fit(X_train,y_train)

y_pred = logreg.predict(X_test)
# Compute and print the confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test,y_pred))


###################### ROC Curve #######################

# Import necessary modules
from sklearn.metrics import roc_curve, roc_auc_score

# Compute predicted probabilities: y_pred_prob
y_pred_prob = logreg.predict_proba(X_test)[:,1]

# Generate ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

# Plot ROC curve
import matplotlib.pyplot as plt
plt.plot([0, 1], [0, 1])
plt.plot(fpr, tpr)
plt.xlabel('1 - Specificity')
plt.ylabel('Sensitivity')
plt.title('ROC Curve')
plt.show()

# Area Under the Curve
roc_auc_score(y_test, y_pred_prob)

########### Log Loss ##################
from sklearn.metrics import log_loss

log_loss(y_test, y_pred_prob)

############# k FOLD CV #############

from sklearn.model_selection import StratifiedKFold
#from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

kfold = StratifiedKFold(n_splits=5, random_state=2021,
                        shuffle=True)

logreg = LogisticRegression()

results = cross_val_score(logreg, X, y, cv=kfold, 
                          scoring='roc_auc')
print(results)
print("ROC AUC: %.4f (%.4f)" % (results.mean(), results.std()))

# Using Accuracy Score
results = cross_val_score(logreg, X, y, cv=kfold)
print(results)
print("Accuracy Score: %.4f (%.4f)" % (results.mean(), results.std()))


