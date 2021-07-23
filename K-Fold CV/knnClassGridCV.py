import pandas as pd

df = pd.read_csv("G:/Statistics (Python)/Datasets/RidingMowers.csv")

dum_df = pd.get_dummies(df)
dum_df = dum_df.drop('Response_Not Bought', axis=1)

from sklearn.neighbors import KNeighborsClassifier

X = dum_df.iloc[:,0:2]
y = dum_df.iloc[:,2]


############### Test for single k at a time ###############
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
kfold = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
model = KNeighborsClassifier(n_neighbors=3)
results = cross_val_score(model, X, y, cv=kfold, 
                          scoring='roc_auc')
print(results)
print("ROC AUC: %.4f (%.4f)" % (results.mean(), results.std()))

############## Testing all Ks at a time ###################

from sklearn.model_selection import GridSearchCV
import numpy as np
parameters = {'n_neighbors': np.array([1,3,5,7,9,11,13,15])}
print(parameters)

from sklearn.model_selection import StratifiedKFold
kfold = StratifiedKFold(n_splits=5, random_state=2021,shuffle=True)

knn = KNeighborsClassifier()
# Tuned according to accuracy score
cv = GridSearchCV(knn, param_grid=parameters,cv=kfold)

cv.fit( X , y )

print(cv.cv_results_  )

# Table of Grid Search CV Results
df_cv = pd.DataFrame(cv.cv_results_  )

print(cv.best_params_)

print(cv.best_score_)

#OR
# Tuned according to AUC score
cv = GridSearchCV(knn, param_grid=parameters,
                  cv=kfold,scoring='roc_auc')
cv.fit( X , y )

print(cv.cv_results_  )

# Table of Grid Search CV Results
df_cv = pd.DataFrame(cv.cv_results_  )

print(cv.best_params_)

print(cv.best_score_)


#OR
# Tuned according to negative log loss score
cv = GridSearchCV(knn, param_grid=parameters,
                  cv=kfold,scoring='neg_log_loss')

cv.fit( X , y )

print(cv.cv_results_  )

# Table of Grid Search CV Results
df_cv = pd.DataFrame(cv.cv_results_  )

print(cv.best_params_)

print(cv.best_score_)


