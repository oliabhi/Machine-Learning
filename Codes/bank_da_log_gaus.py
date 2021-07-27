import pandas as pd

df = pd.read_csv("G:/Statistics (Python)/Cases/bank/bank-full.csv",sep=";")
dum_df = pd.get_dummies(df,drop_first=True)

X = dum_df.iloc[:,:-1]
y = dum_df.iloc[:,-1]

print(dum_df.y_yes.unique())

# Import the necessary modules
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
#########################LDA####################################

lda = LinearDiscriminantAnalysis()
kfold = StratifiedKFold(n_splits=5, random_state=2021,
                        shuffle=True)
results = cross_val_score(lda, X, y, cv=kfold, 
                          scoring='roc_auc_ovr')
AUCs = results
print(AUCs)
print("Mean AUC: %.2f" % (AUCs.mean()))

#########################QDA####################################

lda = QuadraticDiscriminantAnalysis()
kfold = StratifiedKFold(n_splits=5, random_state=2021,
                        shuffle=True)
results = cross_val_score(lda, X, y, cv=kfold, 
                          scoring='roc_auc_ovr')
AUCs = results
print(AUCs)
print("Mean AUC: %.2f" % (AUCs.mean()))


#########################Gaussian####################################

lda = GaussianNB()
kfold = StratifiedKFold(n_splits=5, random_state=2021,
                        shuffle=True)
results = cross_val_score(lda, X, y, cv=kfold, 
                          scoring='roc_auc_ovr')
AUCs = results
print(AUCs)
print("Mean AUC: %.2f" % (AUCs.mean()))

#########################Logistic Regression#########################

lda = LogisticRegression()
kfold = StratifiedKFold(n_splits=5, random_state=2021,
                        shuffle=True)
results = cross_val_score(lda, X, y, cv=kfold, 
                          scoring='roc_auc_ovr')
AUCs = results
print(AUCs)
print("Mean AUC: %.2f" % (AUCs.mean()))
