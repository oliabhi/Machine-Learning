import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import VotingClassifier
from sklearn.metrics import roc_auc_score

lda = LinearDiscriminantAnalysis()
logreg = LogisticRegression()
qda = QuadraticDiscriminantAnalysis()

Voting = VotingClassifier(estimators=[('LDA',lda),
                                      ('LR',logreg),
                                      ('QDA',qda)],voting='soft')

train = pd.read_csv("F:/Kaggle/Santander Customer/train.csv")

X = train.iloc[:,1:-1]
y = train.iloc[:,-1]


Voting.fit(X,y)


test = pd.read_csv("F:/Kaggle/Santander Customer/test.csv")
X_test = test.iloc[:,1:]

y_pred_prob = Voting.predict_proba(X_test)[:,1]


submit = pd.DataFrame({'ID':test.ID, 'TARGET':y_pred_prob})

submit.to_csv("F:/Kaggle/Santander Customer/submit_vote.csv",
              index=False)

#########################lda #########################
lda.fit(X,y)


test = pd.read_csv("F:/Kaggle/Santander Customer/test.csv")
X_test = test.iloc[:,1:]

y_pred_prob = lda.predict_proba(X_test)[:,1]


submit = pd.DataFrame({'ID':test.ID, 'TARGET':y_pred_prob})

submit.to_csv("F:/Kaggle/Santander Customer/submit_lda.csv",
              index=False)

#########################qda #########################
qda.fit(X,y)


test = pd.read_csv("F:/Kaggle/Santander Customer/test.csv")
X_test = test.iloc[:,1:]

y_pred_prob = qda.predict_proba(X_test)[:,1]


submit = pd.DataFrame({'ID':test.ID, 'TARGET':y_pred_prob})

submit.to_csv("F:/Kaggle/Santander Customer/submit_qda.csv",
              index=False)

######## weighted vote ###############################

Voting = VotingClassifier(estimators=[('LDA',lda),
                                      ('LR',logreg),
                                      ('QDA',qda)],voting='soft',
                          weights=[0.404456462499476,
                                   0.32164717226344697,
                                   0.2738963652370771])



Voting.fit(X,y)


test = pd.read_csv("F:/Kaggle/Santander Customer/test.csv")
X_test = test.iloc[:,1:]

y_pred_prob = Voting.predict_proba(X_test)[:,1]


submit = pd.DataFrame({'ID':test.ID, 'TARGET':y_pred_prob})

submit.to_csv("F:/Kaggle/Santander Customer/submit_wt_vote.csv",
              index=False)