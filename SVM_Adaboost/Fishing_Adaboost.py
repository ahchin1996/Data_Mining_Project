import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import  ensemble, preprocessing, metrics
from sklearn.svm import SVC


data_feature_name = ['SFH','popUpWidnow','SSLfinal_State','Request_URL','URL_of_Anchor','web_traffic','URL_Length','age_of_domain','having_IP_Address','Result']    #設定標籤
data = pd.read_csv('D:\Python\DN_report2\PhishingData.data',header=None,names=data_feature_name,sep = ',')

Y = data['Result']
X = data.drop('Result',axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0, train_size=0.8)
boost = ensemble.AdaBoostClassifier(n_estimators = 100,learning_rate=0.6)
boost_fit = boost.fit(X_train, y_train)
# 預測
test_y_predicted = boost.predict(X_test)
# 績效
accuracy = metrics.accuracy_score(y_test, test_y_predicted)
print(accuracy)