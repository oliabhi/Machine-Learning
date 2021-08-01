import h2o
import numpy as np
h2o.init()

df = h2o.import_file("F:/Kaggle/Santander Customer/train.csv")
df.col_names

df_test = h2o.import_file("F:/Kaggle/Santander Customer/test.csv")
df_test.col_names

y_col = 'TARGET'
x_cols = df.col_names
x_cols.remove(y_col)
x_cols.remove('ID')
print("Response = " + y_col)
print("Pridictors = " + str(x_cols))

df['TARGET'] = df['TARGET'].asfactor()
df['TARGET'].levels()

train, test = df.split_frame(ratios=[.8],seed=2021)

######################## Grid Search ############################################

from h2o.grid.grid_search import H2OGridSearch
hyper_parameters = {'mtries': list(np.arange(10,60,10))}
model_rf = H2ORandomForestEstimator(seed=2021)
gs = H2OGridSearch(model=model_rf,hyper_params=hyper_parameters)
gs.train(x=x_cols, y=y_col, training_frame=train, 
                   validation_frame=test, 
                   model_id="model_rf_tune")
models = gs.get_grid(sort_by='auc', decreasing=True)
best_model = models.models[0]
best_params = best_model.actual_params
best_params
best_model_perf1 = best_model.model_performance(test)
best_model_perf1.auc()

######################################################################################

y_pred = best_model.predict(test_data=df_test)
preds = y_pred.as_data_frame()
TestID = df_test["ID"]
ID = TestID.as_data_frame()
ID = ID['ID']
TARGET = preds['p1']
import pandas as pd
submit = pd.DataFrame({'ID':ID, 'TARGET':TARGET})

submit.to_csv("F:/Kaggle/Santander Customer/submit_h2o_rf.csv",
              index=False)


