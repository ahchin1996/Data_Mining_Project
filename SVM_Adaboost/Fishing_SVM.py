import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


data_feature_name = ['SFH','popUpWidnow','SSLfinal_State','Request_URL','URL_of_Anchor','web_traffic','URL_Length','age_of_domain','having_IP_Address','Result']    #設定標籤
data = pd.read_csv('D:\Python\DN_report2\PhishingData.data',header=None,names=data_feature_name,sep = ',')

# 切分訓練與測試資料
Y = data['Result']
X = data.drop('Result',axis=1)
X=X.to_numpy()
Y = Y.to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0, train_size=0.8)

# Adaboost
clf = SVC(kernel= 'linear',probability=True)
clf.fit(X_train,y_train)
clf.score(X_test,y_test)
print(clf.score(X_test,y_test))