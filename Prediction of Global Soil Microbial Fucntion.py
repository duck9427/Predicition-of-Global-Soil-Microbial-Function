import time
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import scipy.stats
import statsmodels.api as sm
from sklearn.model_selection import cross_val_score, KFold
import forestci as fci
from sklearn.model_selection import ShuffleSplit

alpha = 0.975

# log cosh quantile is a regularized quantile loss function
def log_cosh_quantile(alpha):
    def _log_cosh_quantile(y_true, y_pred):
        err = y_pred - y_true
        err = np.where(err < 0, alpha * err, (1 - alpha) * err)
        grad = np.tanh(err)
        hess = 1 / np.cosh(err)**2
        return grad, hess
    return _log_cosh_quantile

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def ci_t(data, confidence):
    sample_mean = np.mean(data)
    sample_std = np.std(data)
    sample_size = len(data)
    df = len(data) - 1

    alpha = (1 - confidence) / 2
    t_score = scipy.stats.t.isf(alpha, df)

    ME = t_score * sample_std / np.sqrt(sample_size)

    lower_limit = sample_mean - ME
    upper_limit = sample_mean + ME

    return (lower_limit, upper_limit)

def bootstrap_mean(data):
    # Repeat sampling from the data, the sample size is the same as the data, and return the sample mean
    return np.mean(np.random.choice(data, size=len(data)))


def draw_bootstrap(data, times=1):
    # Initialize a null array with length of "times"
    bs_mean = np.empty(times)

    # Conduct multiple ("times") sampling, and store the sample mean value obtained each time in bs_ Mean
    for i in range(times):
        bs_mean[i] = bootstrap_mean(data)

    return bs_mean

def expected_calibration_error(y, proba, bins='fd'):
    import numpy as np
    bin_count, bin_edges = np.histogram(proba, bins=bins)
    n_bins = len(bin_count)

    bin_edges[0] -= 1e-8  # because left edge is not included
    bin_id = np.digitize(proba, bin_edges, righ=True) - 1

    bin_ysum = np.bincount(bin_id, weights=y, minlength=n_bins)
    bin_probasum = np.bincount(bin_id, weights=proba, minlength=n_bins)
    bin_ymean = np.divide(bin_ysum, bin_count, out=np.zeros(n_bins), where=bin_count > 0)
    bin_probamean = np.divide(bin_probasum, bin_count, out=np.zeros(n_bins), where=bin_count > 0)

    ece = np.abs((bin_probamean - bin_ymean) * bin_count).sum() / len(proba)
    return ece


def plotModelResults(model, X_train, X_test, y_train, y_test, plot_intervals=False, plot_anomalies=False):
    """
        Plots modelled vs fact values, prediction intervals and anomalies

    """
    prediction = model.predict(X_test)

    plt.figure(figsize=(15, 7))
    plt.plot(prediction, "g", label="prediction", linewidth=2.0)
    plt.plot(y_test.values, label="actual", linewidth=2.0)

    if plot_intervals:
        cv = cross_val_score(model, X_train, y_train,
                             cv=tscv,
                             scoring="neg_mean_absolute_error")
        # scoring='accuracy'  accuracy：The evaluation index is accuracy, and the default value can be omitted
        # cv：Select the number of folds for each test
        mae = cv.mean() * (-1)
        deviation = cv.std()

        scale = 20
        lower = prediction - (mae + scale * deviation)
        upper = prediction + (mae + scale * deviation)

        plt.plot(lower, "r--", label="upper bond / lower bond", alpha=0.5)
        plt.plot(upper, "r--", alpha=0.5)

        if plot_anomalies:
            anomalies = np.array([np.NaN] * len(y_test))
            anomalies[y_test < lower] = y_test[y_test < lower]
            anomalies[y_test > upper] = y_test[y_test > upper]
            plt.plot(anomalies, "o", markersize=10, label="Anomalies")

    error = mean_absolute_percentage_error(prediction, y_test)
    plt.title("Mean absolute percentage error {0:.2f}%".format(error))
    plt.legend(loc="best")
    plt.tight_layout()
    plt.grid(True);
    plt.savefig("linear.png")


names=['Altitude','MAT','MAP','Depth','N','P','K','pH','Total C','OrganicMatters','TN','TP']
temp_data = pd.read_excel(r'D:\DATA\DATA_HYQ\Features correct.xlsx')
y = temp_data[names[-5:]]  # target columns
# use method ".drop(['column name'],axis=1)" to show that one specific
# feature columns are not included in the calculation
X = temp_data[names[0:-5]]


#Model training
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score  # Divide training set and test set
from sklearn.metrics import mean_absolute_error,r2_score


# Split train and test dataset, in which X represents features and Y represents target varibles
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# print(X_train.shape)
# print(y_train.shape)
# print(X_test.shape)
# print(y_test.shape)
# All data finished preparation
# Store current column name
# features = list(X_train.columns)
# print("Output column names：",features)
# Use default parameter at first
#model = xgb.XGBRegressor(learning_rate = 0.1,
#                         n_estimators = 80,
#                         max_depth = 10,
#                         min_child_weight = 3,
#                         subsample = 0.9,
#                         colsample_bytree = 0.7,
#                         seed = 2)


#normal predict
multi_xgb = MultiOutputRegressor(XGBRegressor(
                                              max_depth=12,
                                              learning_rate=0.1,
                                              n_estimators=136,
                                              min_child_weight=13,
                                              subsample=0.9,
                                              colsample_bytree=0.8,
                                              seed=2
                                              ))
model = multi_xgb.fit(X_train, y_train)
y_pred = multi_xgb.predict(X_test)
print("R2 test:", model.score(X_test, y_test))
print("R2 train:", model.score(X_train, y_train))
print("RMSE:", metrics.mean_squared_error(y_test, y_pred, multioutput='raw_values')**0.05)
MAPE = metrics.mean_absolute_percentage_error(y_test, y_pred, multioutput='raw_values')
print("MAPE:", MAPE)
print("CI:", ci_t(y_pred[:,3], 0.95))
volume = y_pred[:,3]
size = len(volume)
bs_mean = draw_bootstrap(volume, 10000)
plt.hist(bs_mean, bins=27, density=True, stacked=True, rwidth=0.9)
plt.show()
print("95% CI level:", np.percentile(bs_mean, [2.5, 97.5]))
#ece = expected_calibration_error(y_pred, MAPE[1])


#over predict
multi_xgb = MultiOutputRegressor(XGBRegressor(objective=log_cosh_quantile(alpha),
                                              max_depth=12,
                                              learning_rate=0.1,
                                              n_estimators=136,
                                              min_child_weight=13,
                                              subsample=0.9,
                                              colsample_bytree=0.8,
                                              seed=2
                                              ))
model = multi_xgb.fit(X_train, y_train)
y_pred_upper = multi_xgb.predict(X_test)

#under predict
multi_xgb = MultiOutputRegressor(XGBRegressor(objective=log_cosh_quantile(1-alpha),
                                              max_depth=12,
                                              learning_rate=0.1,
                                              n_estimators=136,
                                              min_child_weight=13,
                                              subsample=0.9,
                                              colsample_bytree=0.8,
                                              seed=2
                                              ))
model = multi_xgb.fit(X_train, y_train)
y_pred_lower = multi_xgb.predict(X_test)
y_true = np.array(y_test.values)
#index = res['upper_bound'] < 0
#print(res[res['upper_bound'] < 0])
#print(X_test[index])
#max_length = 350
fig = plt.figure()
plt.plot(list(y_true[:,3]), 'gx', label=u'real value')
plt.plot(y_pred_upper[:,3], 'y_', label=u'Q up')
plt.plot(y_pred_lower[:,3], 'b_', label=u'Q low')
index = np.array(range(0, len(y_pred_upper[:,3])))
plt.fill(np.concatenate([index, index[::-1]]),
         np.concatenate([y_pred_upper[:,3], y_pred_lower[:,3][::-1]]),
         alpha=.5, fc='b', ec='None', label='95% prediction interval')
plt.xlabel('$Data$')
plt.ylabel('$Index$')
plt.legend(loc='upper right')
plt.show()
res = pd.DataFrame({'lower_bound' : y_pred_lower[:,3], 'true': y_true[:,3], 'upper_bound': y_pred_upper[:,3]})
count = res[(res.true >= res.lower_bound) & (res.true <= res.upper_bound)].shape[0]
total = res.shape[0]
#print(f'pref = {count} / {total}')
print('Within CI Percent: {:.2%}'.format(count/total))
#xgb_unbiased = fci.random_forest_error()
#print("ECE:", expected_calibration_error())


# # Feature importance
# importances = list(multi_xgb.feature_importances_)
# print("Feature weight:", importances)
# start_time = time.time()  # Start time of model
# Combine column with importance
# feature_importances = [(feature, round(importance, 3)) for feature, importance in zip(features, importances)]
# print(feature_importances)
# # Sorting
# feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)
# print(feature_importances)

no_est = 0 # index of target you want feature importance for


# get estimator
est = multi_xgb.estimators_[0]


# get feature importances
feature_importances = pd.DataFrame(est.feature_importances_,
                                   columns=['importance']).sort_values('importance')
print(feature_importances)
feature_importances.plot(kind = 'barh')
plt.show()


import os
import joblib

# Creation of file directory
dirs = r'D:\DATA\HYQ_DATA'
if not os.path.exists(dirs):
    os.makedirs(dirs)


# Store the model
joblib.dump(multi_xgb, dirs + '/multi_xgb.pkl')


# Invokation of model
xgb_hyq = joblib.load(dirs+'/multi_xgb.pkl')
test = pd.read_csv(r'D:\DATA\HYQ_DATA\data\2022-126u.csv')
test = test.drop('time',axis=1)
print('Prediction results:\n', xgb_hyq.predict(test))
pred = xgb_hyq.predict(test)
pred = pd.DataFrame(pred)
pred.to_csv(r'D:\DATA\HYQ_DATA\result\2022-126.csv',index=False,header=['resistance','tolerance','carbon fixation','carbon degradation','nitrogen fixation','nitrogen degradation'])


# Start parameter tuning
model.get_params


# Use GridSearchCV to select optimal parameters
from sklearn.model_selection import GridSearchCV


# Set selection parameters
param_grid = {
    # 'bootstrap': [True],               # Whether to build a tree by sampling the sample set back树
    # 'max_depth': [1],                  # Maximum depth of decision tree
    # 'max_features': ['auto'],          # The maximum number of features considered when constructing the optimal model of the decision tree. The default is "auto", which means that the maximum number of features is the square root of N
    # 'min_samples_leaf': [20],          # Minimum sample number of leaf nodes
    # 'min_samples_split': [2, 11, 22],  # Minimum number of samples required for internal node subdivision
    # 'n_estimators': [650, 670, 700],
    # 'min_weight_fraction_leaf':[0,0.5],
}
grid_search_xgb = GridSearchCV(estimator=XGBRegressor(random_state=0),
                              param_grid=param_grid, scoring='neg_mean_squared_error',
                              cv=5)
grid_search_xgb.fit(X_train, y_train)


# Model Storation
print(grid_search_xgb.best_params_)


model1 = XGBRegressor(learning_rate = 0.1,
                      n_estimators = 300,
                      max_depth = 7,
                      min_child_weight = 3,
                      subsample = 1.0,
                      colsample_bytree = 0.8,
                      seed = 0)
model1.fit(X_train, y_train)         # Train model
y_predict1 = model1.predict(X_test)  # Divided test dataset before making predictions


cv_params = {'min_child_weight': [1,2,3,4,5,6,7,8,9]}
other_params = {'learning_rate': 0.1, 'n_estimators': 200, 'max_depth': 4, 'min_child_weight': 4, 'seed': 0,
                'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1}
model = xgb.XGBRegressor(**other_params)
optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='r2', cv=5, verbose=1, n_jobs=4)
optimized_GBM.fit(X_train, y_train)
evalute_result = optimized_GBM.cv_results_
print('Iteration result:{0}'.format(evalute_result))
print('Best parameters:{0}'.format(optimized_GBM.best_params_))
print('Best model score:{0}'.format(optimized_GBM.best_score_))
error2 = mean_absolute_error(y_test, y_predict1)
error3 = r2_score(y_test, y_predict1)
print('MAE under tuned parameter:', error2)
print('Coefficient of determination under tuned parameters:', error3)
