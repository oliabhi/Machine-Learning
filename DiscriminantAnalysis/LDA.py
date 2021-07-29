import pandas as pd

df = pd.read_csv("G:/Statistics (Python)/Cases/Satellite Imaging/Satellite.csv",sep=";")

X = df.iloc[:,0:36]
y = df.iloc[:,36]
# Label Encoding for multi-class
## Unique classes
print(df.classes.unique())

# from sklearn.preprocessing import LabelEncoder
# le = LabelEncoder()
# y = le.fit_transform(df.iloc[:,36])

# Import the necessary modules
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, 
                                                    random_state=2020, stratify=y)

da = LinearDiscriminantAnalysis()
da.fit(X_train,y_train)
y_pred = da.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test,y_pred))

from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
plot_confusion_matrix(da,X_test,y_test,labels=df.classes.unique())
plt.show()

y_pred_prob = da.predict_proba(X_test)

from sklearn.metrics import roc_auc_score,log_loss
roc_auc_score(y_test,y_pred_prob,multi_class='ovr')
log_loss(y_test,y_pred_prob)
