import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.utils import shuffle
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, KFold, train_test_split

data = pd.read_csv(r'data\data.csv')
num = data.shape[0]
data = shuffle(data,n_samples=num)

total_error = 0
total_sepsis_error = 0
total_negative_error = 0

train, test = train_test_split(data, train_size=0.2)
feature_columns = ['ed_hour','age','gender','race','arrival_transport','temperature','heartrate','resprate','o2sat','sbp','dbp','map','shock index','acuity']
target_column = 'label'
train_x = train[feature_columns].values
train_y = train[target_column].values
test_x = test[feature_columns].values
test_y = test[target_column].values

cv_params = {'learning_rate': [0.01, 0.05, 0.07, 0.1, 0.2]}
other_params = {'learning_rate': 0.1, 'n_estimators': 100, 'max_depth': 5, 'min_child_weight': 3, 'seed': 0,
                    'subsample': 0.6, 'colsample_bytree': 0.7, 'gamma': 0.3, 'reg_alpha': 1, 'reg_lambda': 2}
model = xgb.XGBClassifier(**other_params)
optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='roc_auc', cv=20, verbose=1, n_jobs=4)
optimized_GBM.fit(train_x, train_y)
evalute_result = optimized_GBM.cv_results_
#print('每轮迭代运行结果:{0}'.format(evalute_result))
print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))

