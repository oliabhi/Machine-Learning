import pandas as pd
import numpy as np

df = pd.read_csv("G:/Statistics (Python)/Cases/Sonar/Sonar.csv")
dum_df = pd.get_dummies(df, drop_first=True)

X = dum_df.iloc[:,:-1]
y = dum_df.iloc[:,-1]

# Import the necessary modules
from sklearn.tree import DecisionTreeClassifier

################################################################
import graphviz 
from sklearn import tree

#######################Grid Search CV###########################
depth_range = [3,4,5,6,7,8,9]
minsplit_range = [5,10,20,25,30]
minleaf_range = [5,10,15]

parameters = dict(max_depth=depth_range,
                  min_samples_split=minsplit_range, 
                  min_samples_leaf=minleaf_range)

from sklearn.model_selection import StratifiedKFold
kfold = StratifiedKFold(n_splits=5, random_state=2021,
                        shuffle=True)

from sklearn.model_selection import GridSearchCV
clf = DecisionTreeClassifier(random_state=2021)
cv = GridSearchCV(clf, param_grid=parameters,
                  cv=kfold,scoring='roc_auc')

cv.fit(X,y)

# Viewing all parameter sets
df_cv = pd.DataFrame(cv.cv_results_)

# Best Parameters
print(cv.best_params_)

print(cv.best_score_)

best_model = cv.best_estimator_
from sklearn import tree
dot_data = tree.export_graphviz(best_model, out_file=None, 
                         feature_names=X_train.columns,  
                         class_names=['Benign','Malignant'],  
                         filled=True, rounded=True,  
                         special_characters=True)  
graph = graphviz.Source(dot_data)  
graph 

