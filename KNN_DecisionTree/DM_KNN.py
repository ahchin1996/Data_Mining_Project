import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("D:\Python\DM_report1\default_of_credit_card_clients.csv")
data = data.drop('ID', axis=1)
data_feature_name = data.columns[0:-1]
Y = data['default_payment_next_month']
X = data.drop('default_payment_next_month', axis=1)
X = data[data_feature_name]

X_train, X_test, y_train, y_test = train_test_split(X, Y,random_state=0, train_size=0.8)


result = []
y_test_pre_list = []
# 建立KNN分類氣
for k in range(1,100):
    clf = KNeighborsClassifier(n_neighbors=k)
    credit_clf = clf.fit(X_train, y_train)
    test_y_pre= clf.predict(X_test)
    acc = clf.score(X_test,y_test)
    result.append(acc)
    y_test_pre_list.append(test_y_pre)
    print('K:%d Acc:%5f'%(k,acc))

max_result_index = result.index(max(result))
y_test_pre_value = y_test_pre_list[max_result_index]

print("Accuracy:"+str(max(result)))
print("K-value:"+str(max_result_index+1))

result = np.array(result)
result = np.reshape(result,(-1,99))
plt.plot(np.mean(result,axis= 0))