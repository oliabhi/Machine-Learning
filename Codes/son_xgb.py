import pandas as pd

df = pd.read_csv("G:/Statistics (Python)/Cases/Sonar/Sonar.csv")

dum_df = pd.get_dummies(df,drop_first=True)

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import cross_val_score

from sklearn.ensemble import BaggingClassifier

X = dum_df.iloc[:,:-1]
y = dum_df.iloc[:,-1]

# Default: Tree Classifier
model_bg = BaggingClassifier(random_state=2021,oob_score=True,
                             max_features=X.shape[1],
                             n_estimators=15,
                             max_samples=int(X.shape[0]*0.8))


######## Bagging ##########                            

kfold = StratifiedKFold(n_splits=5, random_state=2021,
                        shuffle=True)
# cross_val_score()
## 1. Fits the model for every fold combination
## 2. Evaluates the model fitted for every fold combination
results = cross_val_score(model_bg, X, y, cv=kfold, 
                          scoring='roc_auc')
print(results)
print("ROC AUC: %.4f (%.4f)" % (results.mean(), results.std()))


##################Tunning XGB using Grid Search CV#####################
from xgboost import XGBClassifier
lr_range = [0.001,0.01,0.2,0.5,0.6,1]
n_est_range = [30,70,100,120,150]
depth_range = [3,4,5,6,7,8,9]


parameters = dict(learning_rate=lr_range,
                  n_estimators=n_est_range,
                  max_depth=depth_range)


from sklearn.model_selection import StratifiedKFold
kfold = StratifiedKFold(n_splits=5, random_state=2021,shuffle=True)

from sklearn.model_selection import GridSearchCV
clf = XGBClassifier(random_state=2021)
cv = GridSearchCV(clf, param_grid=parameters,
                  cv=kfold,scoring='roc_auc')

cv.fit(X,y)
df_cv = pd.DataFrame(cv.cv_results_)
print(cv.best_params_)

print(cv.best_score_)
