import pandas as pd
import numpy as np

df = pd.read_csv("G:\\Statistics (Python)\\Datasets\\iris.csv")

X = df.iloc[:,:4]
y = df['Species'].astype('category')

# Import the necessary modules
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, 
                                                    random_state=2018,
                                                    stratify=y)

clf = DecisionTreeClassifier(random_state=2021)
clf = clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test,y_pred))

######################################
import graphviz 
#dot_data = tree.export_graphviz(clf, out_file=None) 
#graph = graphviz.Source(dot_data) 
#graph 

from sklearn import tree
dot_data = tree.export_graphviz(clf, out_file=None, 
                         feature_names=X_train.columns,  
                         class_names=['setosa', 'versicolor', 'virginica'],  
                         filled=True, rounded=True,  
                         special_characters=True)  
graph = graphviz.Source(dot_data)  
graph 

#######################Grid Search CV###########################
depth_range = [3,4,5,6,7,8,9]
minsplit_range = [5,10,20,25,30]
minleaf_range = [5,10,15]

parameters = dict(max_depth=depth_range,
                  min_samples_split=minsplit_range, 
                  min_samples_leaf=minleaf_range)

from sklearn.model_selection import GridSearchCV
clf = DecisionTreeClassifier(random_state=2018)


from sklearn.model_selection import StratifiedKFold
kfold = StratifiedKFold(n_splits=5, random_state=2020,shuffle=True)

cv = GridSearchCV(clf, param_grid=parameters,
                  cv=kfold)

cv.fit(X,y)
# Best Parameters
print(cv.best_params_)

print(cv.best_score_)

best_model = cv.best_estimator_
from sklearn import tree
dot_data = tree.export_graphviz(best_model, out_file=None, 
                         feature_names=X_train.columns,  
                         class_names=['setosa', 'versicolor', 'virginica'],  
                         filled=True, rounded=True,  
                         special_characters=True)  
graph = graphviz.Source(dot_data)  
graph 
########################################################
import matplotlib.pyplot as plt

best_model.feature_importances_

ind = np.arange(4)
plt.bar(ind,best_model.feature_importances_)
plt.xticks(ind,(X.columns))
plt.title('Feature Importance')
plt.xlabel("Variables")
plt.show()
#######################################################



