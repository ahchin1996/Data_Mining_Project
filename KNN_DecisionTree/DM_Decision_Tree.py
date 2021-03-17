import numpy as np
import pandas as pd
import pydotplus
from sklearn import tree
from sklearn.model_selection import train_test_split
import os
from sklearn.metrics import accuracy_score

data = pd.read_csv("D:\Python\DM_report1\default_of_credit_card_clients.csv")
# data.head()
# print(data.head())
# data.isnull()
# data.columns

#提取訓練集與測試集
data = data.drop('ID', axis=1)
data_feature_name = data.columns[0:-1]
Y = data['default_payment_next_month']
X = data.drop('default_payment_next_month', axis=1)
X = data[data_feature_name]
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0, train_size=0.8)

#建構決策數模型
model_tree = tree.DecisionTreeClassifier(max_depth=5)
model_tree.fit(X_train, y_train)

# 評論模型準確度
y_prob = model_tree.predict(X_test)
a = accuracy_score(y_test,y_prob)
print(a)

#可視化
data_ = pd.read_csv("D:\Python\DM_report1\default_of_credit_card_clients.csv")
data_ = data_.drop('ID', axis=1)
data_feature_name = data_.columns[0:-1]
data_target_name = np.unique(data_["default_payment_next_month"])
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
graph = tree.export_graphviz(model_tree,out_file=None,feature_names=data_feature_name,
                             class_names=['1','0'],filled=True, rounded=True,special_characters=True,max_depth=5)
graph = pydotplus.graph_from_dot_data(graph)
graph.write_pdf('tree4.pdf')

name = ['y_test',"y_prob"]
test = pd.DataFrame(columns=name)
test['y_test']=y_test
test["y_prob"]=y_prob
print(test)
test.to_csv('D:\Python\DM_report1\ tree_y_output.csv',encoding='gbk')