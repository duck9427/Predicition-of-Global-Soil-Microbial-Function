import os
import joblib
import pandas as pd

# 创建文件目录
dirs = r'D:\DATA\HYQ_DATA'
if not os.path.exists(dirs):
    os.makedirs(dirs)


#读取模型
xgb_hyq = joblib.load(dirs+'/multi_xgb.pkl')
test = pd.read_csv(r'D:\DATA\HYQ_DATA\Uncertainty of model\Tropic.csv')
test = test.drop('CONTINENT',axis=1)
test = test.drop('latitude',axis=1)
test = test.drop('longitude',axis=1)
print('Prediction results:\n', xgb_hyq.predict(test))
pred = xgb_hyq.predict(test)
pred = pd.DataFrame(pred)
pred.to_csv(r'D:\DATA\HYQ_DATA\Uncertainty\Tropic.csv',index=False,
            header=['resistance','tolerance','carbon fixation','carbon degradation',
                    'nitrogen fixation','nitrogen degradation'])