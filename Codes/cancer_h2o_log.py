import h2o
h2o.init()

df = h2o.import_file("G:/Statistics (Python)/Cases/Wisconsin/BreastCancer.csv")
#df.summary()
df.col_names

y_col = 'Class'
x_cols = df.col_names[1:-1]
#x.remove(y)
#x.remove('ID')
print("Response = " + y_col)
print("Pridictors = " + str(x_cols))

df['Class'] = df['Class'].asfactor()
df['Class'].levels()

train,  test = df.split_frame(ratios=[.8],seed=2021)
print(df.shape)
print(train.shape)
#print(valid.shape)
print(test.shape)

from h2o.estimators.glm import H2OGeneralizedLinearEstimator
glm_logistic = H2OGeneralizedLinearEstimator(family = "binomial")
glm_logistic.train(x=x_cols, y=y_col, training_frame=train, 
                   validation_frame=test, model_id="glm_logistic")

y_pred = glm_logistic.predict(test_data=test)

y_pred_df = y_pred.as_data_frame()

print(glm_logistic.auc() )
print(glm_logistic.confusion_matrix() )

#### Gaussian NB
from h2o.estimators.naive_bayes import H2ONaiveBayesEstimator
model_nb = H2ONaiveBayesEstimator()
model_nb.train(x=x_cols, y=y_col, training_frame=train, 
                   validation_frame=test, model_id="model_nb")

y_pred = model_nb.predict(test_data=test)

y_pred_df = y_pred.as_data_frame()

print(model_nb.auc() )
print(model_nb.confusion_matrix() )

##### Random Forest 

from h2o.estimators.random_forest import H2ORandomForestEstimator

model_rf = H2ORandomForestEstimator(seed=2021)
model_rf.train(x=x_cols, y=y_col, training_frame=train, 
                   validation_frame=test, model_id="model_rf")

y_pred = model_rf.predict(test_data=test)

y_pred_df = y_pred.as_data_frame()

print(model_rf.auc() )
print(model_rf.confusion_matrix() )

#### Grid Search #####
from h2o.grid.grid_search import H2OGridSearch
hyper_parameters = {'mtries': [3,4,5,6]}
model_rf = H2ORandomForestEstimator(seed=2021)
gs = H2OGridSearch(model=model_rf,hyper_params=hyper_parameters)
gs.train(x=x_cols, y=y_col, training_frame=train, 
                   validation_frame=test, 
                   model_id="model_rf_tune")
models = gs.get_grid(sort_by='auc', decreasing=True)
best_model = models.models[0]
best_params = best_model.actual_params
print(best_params)
best_model_perf1 = best_model.model_performance(test)
best_model_perf1.auc()

h2o.cluster().shutdown()
