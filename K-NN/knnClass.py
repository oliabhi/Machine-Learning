import numpy as np
import pandas as pd

df = pd.read_csv("G:/Statistics (Python)/Datasets/RidingMowers.csv")

dum_df = pd.get_dummies(df)
dum_df = dum_df.drop('Response_Not Bought', axis=1)

from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, accuracy_score
from sklearn.neighbors import KNeighborsClassifier

X = dum_df.iloc[:,0:2]
y = dum_df.iloc[:,2]

y.value_counts()
pd.crosstab(index=y,columns='Prop',normalize='all')

#### Visualizing the Data
import matplotlib.pyplot as plt
X_B = X[y==1]
X_NB = X[y==0]
plt.scatter(X_B.Income,X_B.Lot_Size,c="green",label="Bought")
plt.scatter(X_NB.Income,X_NB.Lot_Size,c="red",label="Not Bought")
plt.legend()
plt.title("Riding Mowers")
plt.xlabel('Income')
plt.ylabel('Lot Size')
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.3, 
                                                    random_state=2021,
                                                    stratify=y)


knn = KNeighborsClassifier(n_neighbors=5)

knn.fit( X_train , y_train )
y_pred = knn.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

########## Supported from version 0.22 onwards ##########
from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(knn,X_test,y_test,labels=[0,1])
plt.show()


#########################################################
from sklearn.metrics import roc_curve, roc_auc_score

# Compute predicted probabilities: y_pred_prob
y_pred_prob = knn.predict_proba(X_test)[:,1]

# Generate ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

############ Plot ROC curve #############
import matplotlib.pyplot as plt
plt.plot([0, 1], [0, 1], 'r:')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()
#########################################
roc_auc_score(y_test, y_pred_prob)
