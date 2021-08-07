import pandas as pd

df = pd.read_csv("G:/Statistics (Python)/Cases/Sonar/Sonar.csv")

dum_df = pd.get_dummies(df,drop_first=True)


X = dum_df.iloc[:,:-1]
y = dum_df.iloc[:,-1]

# Import the necessary modules
from sklearn.model_selection import train_test_split 

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, 
                                                    random_state=2021,
                                                    stratify=y)

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,QuadraticDiscriminantAnalysis
lda = LinearDiscriminantAnalysis()
qda = QuadraticDiscriminantAnalysis()

from sklearn.naive_bayes import GaussianNB
gaussian = GaussianNB()

from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(random_state=2021)

from sklearn.ensemble import StackingClassifier
models_considered = [('Logistic Regression', logreg),
                     ('LDA', lda),('QDA',qda),('Naive Bayes',gaussian),
                     ('Decision Tree',dtc)]

from xgboost import XGBClassifier
clf = XGBClassifier(random_state=2021)

stack = StackingClassifier(estimators = models_considered,
                           final_estimator=clf,
                           stack_method="predict_proba")

stack.fit(X_train,y_train)

############ w/o passthrough ##################
y_pred_prob = stack.predict_proba(X_test)[:,1]

from sklearn.metrics import roc_auc_score
print(roc_auc_score(y_test, y_pred_prob))

########### with passthrough ##################
stack = StackingClassifier(estimators = models_considered,
                           final_estimator=clf,
                           stack_method="predict_proba",
                           passthrough=True)

stack.fit(X_train,y_train)

y_pred_prob = stack.predict_proba(X_test)[:,1]

from sklearn.metrics import roc_auc_score
print(roc_auc_score(y_test, y_pred_prob))




