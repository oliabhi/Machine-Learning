from sklearn.datasets import load_wine
df_tuple = load_wine(return_X_y=True, as_frame=True)

X = df_tuple[0]
y = df_tuple[1]

########################### Tunning Linear #####################################

########################## Grid Search CV #############################
from sklearn.model_selection import GridSearchCV

C_range = np.linspace(0.001,2)
decision_shape = ['ovo','ovr']
param_grid = dict( C=C_range, decision_function_shape=decision_shape)

from sklearn.model_selection import StratifiedKFold
kfold = StratifiedKFold(n_splits=5, random_state=2020,shuffle=True)
svmGrid = GridSearchCV(SVC(probability=True,kernel='linear'), 
                       param_grid=param_grid, cv=kfold, 
                       scoring="neg_log_loss",verbose=3)
svmGrid.fit(X, y)

# Best Parameters
print(svmGrid.best_params_)
#print(svmGrid.cv_results_)
print(svmGrid.best_score_)

############################## Tunning for RBF ###############@@#########
from sklearn.model_selection import GridSearchCV

C_range = np.linspace(0.01,4,10)
decision_shape = ['ovo','ovr']
gamma_range = np.linspace(0.001,10,10)

parameters = dict(gamma=gamma_range, C=C_range,
                  decision_function_shape=decision_shape)

svc = SVC(probability=True,kernel='rbf')

from sklearn.model_selection import StratifiedKFold
kfold = StratifiedKFold(n_splits=5, random_state=2021,shuffle=True)
svmGrid = GridSearchCV(svc, param_grid=parameters, cv=kfold,
                       scoring='neg_log_loss',verbose=3)
svmGrid.fit(X, y)

# Best Parameters
print(svmGrid.best_params_)

print(svmGrid.best_score_)