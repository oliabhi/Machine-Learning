import pandas as pd
import numpy as np
df = pd.read_csv("F:/Kaggle/Santander Customer/train.csv")
df=df.drop(['ID'],axis=1)

X = df.iloc[:,:-1]
y = df.iloc[:,-1]

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
Xscaled=scaler.transform(X)

from sklearn.decomposition import PCA
pca = PCA()
principalComponents = pca.fit_transform(Xscaled)
print(pca.explained_variance_)
print(pca.explained_variance_ratio_) 
print(pca.explained_variance_ratio_ * 100) 
#print(pca.singular_values_)  

import matplotlib.pyplot as plt
ys = pca.explained_variance_ratio_ * 100
xs = np.arange(1,370)
plt.plot(xs,ys)
plt.show()

import matplotlib.pyplot as plt
ys = np.cumsum(pca.explained_variance_ratio_ * 100)
xs = np.arange(1,370)
plt.plot(xs,ys)
plt.show()

# PCs upto 90% variation explained
np.sum(ys<=91)

# principalComponents are PCA scores
cols = []
for i in range(1,76):
    name =  "PC" + str(i)
    cols.append(name)

X_PCs = pd.DataFrame(principalComponents[:,:75],
                 columns = cols)

####################Fit a ML Model##############################
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_PCs,y)
###################### Test Set ########################
df_test = pd.read_csv("F:/Kaggle/Santander Customer/test.csv", header='infer')
Orig_X = df_test.iloc[:,1:]

Orig_X_scaled = scaler.transform(Orig_X)
princ_comp_test = pca.transform(Orig_X_scaled)
    
X_test_PCs = pd.DataFrame(princ_comp_test[:,:75],
                 columns = cols)    

y_pred_prob = model.predict_proba(X_test_PCs)[:,1]

submit = pd.DataFrame({'ID':df_test["ID"],
                       'TARGET':y_pred_prob})
submit.to_csv("F:/Kaggle/Santander Customer/submit_PCA_glm75.csv",index=False)

