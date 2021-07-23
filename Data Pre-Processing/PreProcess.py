import pandas as pd

iris = pd.read_csv("F:/ML with Python/Datasets/iris.csv")

dum_iris = pd.get_dummies(iris)
dum_iris = pd.get_dummies(iris,drop_first=True)


cars = pd.read_csv("F:/ML with Python/Datasets/Cars93.csv")

cars.head(n=10)

cars = cars.set_index('Model')
# OR
cars.set_index('Model',inplace=True)

dum_cars = pd.get_dummies(cars, drop_first=True)

dum_cars.head(n=10)

## Label Encoding
from sklearn.preprocessing import LabelEncoder
lbcode = LabelEncoder()
y = ['a','b','a','a','c','a','b','b','a','c','a']

trny = lbcode.fit_transform(y)
print(trny)

carsMissing = pd.read_csv("F:/ML with Python/Datasets/Cars93Missing.csv",index_col=1)

carsMissing.shape

carsDropNA = carsMissing.dropna()
carsDropNA.shape

# Dummying the data
dum_cars_miss = pd.get_dummies(cars, drop_first=True)

from sklearn.impute import SimpleImputer
imp = SimpleImputer(strategy='mean')
carsImputed = imp.fit_transform(dum_cars_miss)

df_carsImputed = pd.DataFrame(carsImputed,
                              columns= dum_cars_miss.columns,
                              index=dum_cars_miss.index)

dum_cars_miss.shape
carsImputed.shape
df_carsImputed.shape

job = pd.read_csv("G:/Statistics (Python)/Datasets/JobSalary2.csv")
impJob = SimpleImputer(strategy='constant',fill_value=0)
impJob.fit(job)
trn_job = impJob.transform(job)
trn_job

#OR

trn_job = impJob.fit_transform(job)
trn_job


import numpy as np
milk = pd.read_csv("F:/ML with Python/Datasets/milk.csv",index_col=0)
milk.head()
np.mean(milk), np.std(milk)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(milk)
milkscaled=scaler.transform(milk)

# OR
milkscaled=scaler.fit_transform(milk)

np.mean(milkscaled[:,0]), np.std(milkscaled[:,0])
np.mean(milkscaled[:,1]), np.std(milkscaled[:,1])
np.mean(milkscaled[:,2]), np.std(milkscaled[:,2])
np.mean(milkscaled[:,3]), np.std(milkscaled[:,3])
np.mean(milkscaled[:,4]), np.std(milkscaled[:,4])

# Converting numpy array to pandas
df_milk = pd.DataFrame(milkscaled,columns=milk.columns,
                       index=milk.index)

from sklearn.preprocessing import MinMaxScaler
minmax = MinMaxScaler()
minmax.fit(milk)
minmaxMilk = minmax.transform(milk)
minmaxMilk[1:5,]

# OR
minmaxMilk = minmax.fit_transform(milk)
# Converting numpy array to pandas
df_milk = pd.DataFrame(minmaxMilk,columns=milk.columns,
                       index=milk.index)