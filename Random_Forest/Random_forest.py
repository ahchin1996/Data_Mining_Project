import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime

train_hire_stats = pd.read_csv("D:/Python/DM_report4/taxi_data/train_hire_stats.csv")
test_hire_stats = pd.read_csv("D:/Python/DM_report4/taxi_data/test_hire_stats.csv")
data_feature_name = train_hire_stats.columns[0:-1]

# train_hire_stats日期轉星期
date =  train_hire_stats['Date']
for i in range(len(date)+1):
    ch = date[i]
    date[i] = datetime.strptime(ch, "%Y-%m-%d").weekday()
    
train_hire_stats['Date'] = date

# test_hire_stats日期轉星期
date =  test_hire_stats['Date']
for i in range(len(date)+1):
    ch = date[i]
    date[i] = datetime.strptime(ch, "%Y-%m-%d").weekday()
test_hire_stats['Date'] = date


test_hire_stats = test_hire_stats.drop("Hire_count",axis=1)
test_hire_stats = test_hire_stats.drop('Test_ID',axis=1)

train_features = train_hire_stats.drop('Hire_count', axis=1)
train_labels = train_hire_stats['Hire_count']
# 使用隨機森林演算法(1000顆樹)
rf = RandomForestRegressor(n_estimators = 1000,random_state = 42)
rf.fit(train_features, train_labels)
# 進行預測
# Use the forest's predict method on the test data
predictions = rf.predict(test_hire_stats)

name = ['predictions']
new_predictions = pd.DataFrame(columns=name)
new_predictions['predictions'] = predictions
new_predictions.to_csv('D:\Python\DM_report4\ new_predictions.csv',encoding='gbk')