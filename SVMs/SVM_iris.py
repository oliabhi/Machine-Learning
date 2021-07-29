import pandas as pd
import numpy as np

df = pd.read_csv("G:/Statistics (Python)/Datasets/iris.csv")

X = df.iloc[:,:4]
y = df.iloc[:,4]

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from sklearn.svm import SVC

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.3, 
                                                    random_state=42,
                                                    stratify=y)

svc = SVC(probability = True,kernel='rbf')
fitSVC = svc.fit(X_train, y_train)
y_pred = fitSVC.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

print(accuracy_score(y_test, y_pred))



############################## Tunning for RBF ########################
from sklearn.model_selection import GridSearchCV

C_range = np.linspace(0.01,4,10)
#C_range = np.array([0.01,0.05,0.1,1,1.5,1.7,2,4])
gamma_range = np.linspace(0.001, 3,10)
#gamma_range = np.array([0.01,0.05,0.1,1,1.5,1.7,2,4])
df_range = ['ovo','ovr']

parameters = dict(gamma=gamma_range, C=C_range,
                  decision_function_shape=df_range)
#cv = StratifiedShuffleSplit(n_splits=5, train_size=2, test_size=None, random_state=42)
svc = SVC(probability=True)

from sklearn.model_selection import StratifiedKFold
kfold = StratifiedKFold(n_splits=5, random_state=2021,shuffle=True)
svmGrid = GridSearchCV(svc, param_grid=parameters, cv=kfold,verbose=3)
svmGrid.fit(X, y)

# Best Parameters
print(svmGrid.best_params_)

print(svmGrid.best_score_)

########################## Random Grid Search CV #################### 
from sklearn.model_selection import RandomizedSearchCV

rcv = RandomizedSearchCV(SVC(probability=True,decision_function_shape='ovo'),
                        random_state=2021,
                        param_distributions=parameters ,
                        cv=kfold,n_iter=10)

rcv.fit( X , y)

print(rcv.best_params_)

print(rcv.best_score_)  

print(rcv.best_estimator_)


############ Tunning for Sigmoid ###################

C_range = np.array([0.01,0.05,0.1,1,1.5,1.7,2,4])
# OR
C_range = np.linspace(0.01,4)
Coef0_range = np.array([0.01,0.05,0.1,1,1.5,1.7,2,4])

parameters = dict(C=C_range,coef0 = Coef0_range )

svc = SVC(probability=True, kernel='sigmoid')
from sklearn.model_selection import StratifiedKFold
kfold = StratifiedKFold(n_splits=5, random_state=42,shuffle=True)

svmGrid = GridSearchCV(svc, param_grid=parameters, cv=kfold)
svmGrid.fit(X, y)

# Best Parameters
print(svmGrid.best_params_)

print(svmGrid.best_score_)

