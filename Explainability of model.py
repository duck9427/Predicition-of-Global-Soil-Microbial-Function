import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt; plt.style.use('seaborn')


data = pd.read_csv(r'D:\DATA\HYQ_DATA\2021-2-SHAP.csv')
data = data.dropna(axis=0)
cols = ['Altitude','MAT','MAP','Depth','N','P','K','pH','Total C','OrganicMatters','TN','TP']
pres = ['resistance']

X = data[cols].values
y = data[pres].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# Train xgboost regression model
model = xgb.XGBRegressor(max_depth=12,
                         learning_rate=0.1,
                         n_estimators=136,
                         min_child_weight=13,
                         subsample=0.9,
                         colsample_bytree=0.8,
                         seed=2)
model.fit(X_train, y_train)
print(r2_score(y_test, model.predict(X_test)))


#SHAP
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(data[cols])
shap_values2 = explainer(data[cols])
print(shap_values.shape)
y_base = explainer.expected_value
print(y_base)
data['pred'] = model.predict(data[cols])
print(data['pred'].mean())
j = 530
pr_explainer = pd.DataFrame()
pr_explainer['feature'] = cols
pr_explainer['feature_value'] = data[cols].iloc[j].values
pr_explainer['shap_value'] = shap_values[j]
#pr_explainer.sort_values('shap_value', ascending=False)

print('y_base + sum_of_shap_values: %.2f'%(y_base + pr_explainer['shap_value'].sum()))
print('y_pred: %.2f'%(data['pred'].iloc[j]))


shap.initjs()
shap.force_plot(explainer.expected_value, shap_values[j], data[cols].iloc[j], matplotlib=True)
shap.plots.beeswarm(shap_values2)
shap.plots.waterfall(explainer.expected_value[30], shap_values[30], data[cols])
shap.plots.bar(shap_values2, show_data=True)
shap.plots.bar(shap_values2[1], show_data=True)
clustering = shap.utils.hclust(data[cols], data['temperature'])
shap.plots.bar(shap_values2,
               clustering=clustering,
               clustering_cutoff=0.5)

shap.dependence_plot('Altitude', shap_values, data[cols], interaction_index=None)
shap.dependence_plot('MAT', shap_values, data[cols], interaction_index=None)
shap.dependence_plot('MAP', shap_values, data[cols], interaction_index=None)
shap.dependence_plot('Depth', shap_values, data[cols], interaction_index=None)
shap.dependence_plot('N', shap_values, data[cols], interaction_index=None)
shap.dependence_plot('P', shap_values, data[cols], interaction_index=None)
shap.dependence_plot('K', shap_values, data[cols], interaction_index=None)
shap.dependence_plot('pH', shap_values, data[cols], interaction_index=None)
shap.dependence_plot('Total C', shap_values, data[cols], interaction_index=None)
shap.dependence_plot('OrganicMatters', shap_values, data[cols], interaction_index=None)
shap.dependence_plot('TN', shap_values, data[cols], interaction_index=None)
shap.dependence_plot('TP', shap_values, data[cols], interaction_index=None)


shap_interaction_values = shap.TreeExplainer(model).shap_interaction_values(data[cols])
shap.summary_plot(shap_interaction_values, data[cols], max_display=12)



out_arr = shap_values
out_df = pd.DataFrame(out_arr,columns =['Altitude','MAT','MAP','Depth','N','P','K','pH','Total C','OrganicMatters','TN','TP'])
out_df.to_excel(r'D:\DATA\HYQ_DATA\SHAP2021\SHAP_RESISTANCE.xlsx', index=False)
shap.dependence_plot('MAT', shap_values, data[cols], interaction_index='OrganicMatters')