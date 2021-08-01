import os
import pandas as pd
import numpy as np

os.chdir("F:/Kaggle/Otto Product Classification")

train = pd.read_csv("train.csv",index_col=0)
test  = pd.read_csv("test.csv",index_col=0)

print(train["target"].unique())

X = train.iloc[:,:-1]
y = train.iloc[:,-1]
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()

logreg.fit(X,y)
y_ans = logreg.predict_proba(test)

sampsub = pd.read_csv("sampleSubmission.csv")
submit = pd.DataFrame(y_ans,index=test.index,
                      columns=sampsub.columns[1:]) 
submit.to_csv("submit_logreg.csv")

###################  Linear  ######################

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis()

lda.fit(X,y)
y_ans = lda.predict_proba(test)

sampsub = pd.read_csv("sampleSubmission.csv")
submit = pd.DataFrame(y_ans,index=test.index,
                      columns=sampsub.columns[1:]) 
submit.to_csv("submit_lda.csv")

###################  Quadratic  ######################

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
qda = QuadraticDiscriminantAnalysis()

qda.fit(X,y)
y_ans = qda.predict_proba(test)

sampsub = pd.read_csv("sampleSubmission.csv")
submit = pd.DataFrame(y_ans,index=test.index,
                      columns=sampsub.columns[1:]) 
submit.to_csv("submit_qda.csv")

################# Decision Tree #########################

from sklearn.tree import DecisionTreeClassifier

depth_range = np.arange(5,35,5)
minsplit_range = np.arange(50,7000,1000)
minleaf_range = np.arange(50,5000,1000)

parameters = dict(max_depth=depth_range,
                  min_samples_split=minsplit_range, 
                  min_samples_leaf=minleaf_range)

from sklearn.model_selection import StratifiedKFold
kfold = StratifiedKFold(n_splits=5, random_state=2021)

from sklearn.model_selection import GridSearchCV
clf = DecisionTreeClassifier(random_state=2018)
cv = GridSearchCV(clf, param_grid=parameters,verbose=3,
                  cv=kfold)

cv.fit(X,y)
# Best Parameters
print(cv.best_params_)

print(cv.best_score_)

best_model = cv.best_estimator_


y_ans = best_model.predict_proba(test)


sampsub = pd.read_csv("sampleSubmission.csv")
submit = pd.DataFrame(y_ans,index=test.index,columns=sampsub.columns[1:]) 

submit.to_csv("submit_DT.csv")
