import pandas as pd

telecom = pd.read_csv("G:/Statistics (Python)/Cases/Telecom/Telecom.csv")

dum_telecom = pd.get_dummies(telecom, drop_first=True)

from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, accuracy_score
from sklearn.naive_bayes import MultinomialNB

X = dum_telecom.iloc[:,0:2]
y = dum_telecom.iloc[:,2]

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.3, 
                                                    random_state=42,
                                                    stratify=y)

multinomial = MultinomialNB()
multinomial.fit(X_train, y_train) # Model Building: Apriori Probs Calculated

y_probs = multinomial.predict_proba(X_test) # Posterior Probs Calculated
y_pred = multinomial.predict(X_test) # Applying built on test data

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

# ROC
from sklearn.metrics import roc_curve, roc_auc_score

# Compute predicted probabilities: y_pred_prob
y_probs = multinomial.predict_proba(X_test)
y_pred_prob = y_probs[:,1]

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

