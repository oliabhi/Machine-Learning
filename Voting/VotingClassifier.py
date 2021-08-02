import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import VotingClassifier
from sklearn.metrics import roc_auc_score

dtc = DecisionTreeClassifier(random_state=2021)
logreg = LogisticRegression()
gauss = GaussianNB()

Voting = VotingClassifier(estimators=[('DT',dtc),
                                      ('LR',logreg),
                                      ('GU',gauss)],voting='soft')

df = pd.read_csv("F:\\Statistics\\Cases\\Wisconsin\\BreastCancer.csv")
dum_df = pd.get_dummies(df, drop_first=True)

X = dum_df.iloc[:,1:10]
y = dum_df.iloc[:,10]

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, 
                                                    random_state=2018,
                                                    stratify=y)

Voting.fit(X_train,y_train)


y_pred = Voting.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test,y_pred))
y_pred_prob = Voting.predict_proba(X_test)[:,1]
print(roc_auc_score(y_test, y_pred_prob))
