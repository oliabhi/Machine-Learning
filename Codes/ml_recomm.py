import pandas as pd
import numpy as np
import surprise
import os

os.chdir(r"F:\Kaggle\Datasets\The Movies Dataset")
orig_ratings = pd.read_csv("ratings_small.csv")

orig_ratings.head()
ratings = orig_ratings.iloc[:,:-1]
ratings.head()
ratings.columns = ['uid','iid','rating']

lowest_rating = ratings['rating'].min()
highest_rating = ratings['rating'].max()

print("Ratings range between {0} and {1}".format(lowest_rating,highest_rating))

reader = surprise.Reader(rating_scale = (lowest_rating,highest_rating))
data = surprise.Dataset.load_from_df(ratings,reader)

similarity_options = {'name': 'cosine', 'user_based': True}
# Default k = 40
algo = surprise.KNNBasic(sim_options = similarity_options)
output = algo.fit(data.build_full_trainset())

pred = algo.predict(uid='100',iid='6')
score = pred.est
print(score)

iids = ratings['iid'].unique()
iids100 = ratings.loc[ratings['uid'] == 100 ,'iid']
print("List of iid that uid={0} has rated:".format(100))
print(iids100)

iids_to_predict = np.setdiff1d(iids,iids100)
print("List of iid which uid={0} did not rate(in all {1}) :".format(100,len(iids_to_predict)))
print(iids_to_predict)

### ratings arbitrarily set to 0
testset = [[100,iid,0.] for iid in iids_to_predict]
predictions = algo.test(testset)
predictions[5]


pred_ratings = np.array([pred.est for pred in predictions])

# Finding the index of maximum predicted rating
i_max = pred_ratings.argmax()

# Recommending the item with maximum predicted rating
iid_recommend_most = iids_to_predict[i_max] 
print("Top item to be recommended for user {0} is {1} with predicted rating as {2}".format(100,iid_recommend_most,pred_ratings[i_max]))

# Getting top 10 items to be recommended for uid = 50
import heapq
i_sorted_10 = heapq.nlargest(10, 
                             range(len(pred_ratings)), 
                             pred_ratings.take)
top_10_items = iids_to_predict[i_sorted_10]
print(top_10_items)
print(pred_ratings[i_sorted_10])


############ Tuning ############

from surprise.model_selection import GridSearchCV
param_grid = {'k': np.arange(10,60,5)}

from surprise.model_selection.split import KFold
kfold = KFold(n_splits=5, random_state=2021, shuffle=True)
gs = GridSearchCV(surprise.KNNBasic, param_grid, 
                  measures=['rmse', 'mae'], cv=kfold)

gs.fit(data)
# best RMSE score
print(gs.best_score['rmse'])

# combination of parameters that gave the best RMSE score
print(gs.best_params['rmse'])


# We can now use the algorithm that yields the best rmse:
algo = gs.best_estimator['rmse']
algo.fit(data.build_full_trainset())

######################################

pred = algo.predict(uid='100',iid='207')
score = pred.est
print(score)

iids = ratings['iid'].unique()
iids50 = ratings.loc[ratings['uid'] == 100 ,'iid']
print("List of iid that uid={0} has rated:".format(100))
print(iids50)

iids_to_predict = np.setdiff1d(iids,iids50)
print("List of iid which uid={0} did not rate(in all {1}) :".format(100,len(iids_to_predict)))
print(iids_to_predict)

### ratings arbitrarily set to 0
testset = [[100,iid,0.] for iid in iids_to_predict]
predictions = algo.test(testset)
predictions[5]


pred_ratings = np.array([pred.est for pred in predictions])

# Finding the index of maximum predicted rating
i_max = pred_ratings.argmax()

# Recommending the item with maximum predicted rating
iid_recommend_most = iids_to_predict[i_max] 
print("Top item to be recommended for user {0} is {1} with predicted rating as {2}".format(50,iid_recommend_most,pred_ratings[i_max]))

# Getting top 10 items to be recommended for uid = 50
import heapq
i_sorted_10 = heapq.nlargest(10, range(len(pred_ratings)), pred_ratings.take)
top_10_items = iids_to_predict[i_sorted_10]
print(top_10_items)















