import pandas as pd

df = pd.read_csv("G:/Statistics (Python)/Cases/Wisconsin/BreastCancer.csv")

dum_df = pd.get_dummies(df)
dum_df = dum_df.drop('Class_Benign', axis=1)

from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import BaggingClassifier

X = dum_df.iloc[:,1:10]
y = dum_df.iloc[:,10]

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.3, 
                                                    random_state=2021,
                                                    stratify=y)

# Default: Tree Classifier
model_bg = BaggingClassifier(random_state=2021,oob_score=True,
                             max_features=X_train.shape[1],
                             n_estimators=15,
                             max_samples=X_train.shape[0])

#OR for any other model bagging
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

model_bg = BaggingClassifier(base_estimator = logreg ,
                             random_state=2021,oob_score=True,
                             max_features=X_train.shape[1],
                             n_estimators=15,max_samples=X_train.shape[0])

######## Building the model ##########                            
model_bg.fit( X_train , y_train )

print("Out of Bag Score = " + "{:.4f}".format(model_bg.oob_score_))

y_pred = model_bg.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

from sklearn.metrics import roc_curve, roc_auc_score

# Compute predicted probabilities: y_pred_prob
y_pred_prob = model_bg.predict_proba(X_test)[:,1]

# Generate ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

# Plot ROC curve
import matplotlib.pyplot as plt
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()
roc_auc_score(y_test, y_pred_prob)

####################### Only Logistic Regression ##########################

model = LogisticRegression()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

# Compute predicted probabilities: y_pred_prob
y_pred_prob = model.predict_proba(X_test)[:,1]

# Generate ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

# Plot ROC curve
import matplotlib.pyplot as plt
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()
roc_auc_score(y_test, y_pred_prob)

