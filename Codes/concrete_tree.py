import pandas as pd
import numpy as np

df = pd.read_csv(r"G:\Statistics (Python)\Cases\Concrete Strength\Concrete_Data.csv")


from sklearn.tree import DecisionTreeRegressor
X = df.iloc[:,:-1]
y = df.iloc[:,-1]

#######################Grid Search CV#############################
depth_range = [3,4,5,6,7,8,9]
minsplit_range = [5,10,20,25,30]
minleaf_range = [5,10,15]

parameters = dict(max_depth=depth_range,
                  min_samples_split=minsplit_range, 
                  min_samples_leaf=minleaf_range)

from sklearn.model_selection import KFold
kfold = KFold(n_splits=5, random_state=2021,shuffle=True)
from sklearn.model_selection import GridSearchCV
clf = DecisionTreeRegressor(random_state=2021)
cv = GridSearchCV(clf, param_grid=parameters,
                  cv=kfold,scoring='r2')

cv.fit(X,y)
# Best Parameters
print(cv.best_params_)

print(cv.best_score_)

best_model = cv.best_estimator_


###################################################################
import graphviz 
from sklearn import tree
# =============================================================================
# dot_data = tree.export_graphviz(clf2, out_file=None) 
# graph = graphviz.Source(dot_data) 
# graph.render("Housing") 
# =============================================================================
dot_data = tree.export_graphviz(best_model, out_file=None, 
                         feature_names=list(X),  
                         filled=True, rounded=True,  
                         special_characters=True)  
graph = graphviz.Source(dot_data)  
graph 



######################################################################
best_model = cv.best_estimator_
import matplotlib.pyplot as plt

best_model.feature_importances_

ind = np.arange(X.shape[1])
plt.bar(ind,best_model.feature_importances_)
plt.xticks(ind,(X.columns),rotation=45)
plt.title('Feature Importance')
plt.xlabel("Variables")
plt.show()
########################################################################