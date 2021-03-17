import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


# 正規化

data_feature_name = ['buying','maint','doors','persons','lug_boot','safety','Class_Value']    #設定標籤
data = pd.read_csv('D:\Python\DN_report2\car.data',header=None,names=data_feature_name,sep = ',')
pattern_bying = {'vhigh':4,"high":3,"med":2,"low":1}
data['buying'] = [pattern_bying[x] if x in pattern_bying else x for x in data['buying']]
pattern_maint = {'vhigh':4,'high':3,'med':2,'low':1}
data['maint'] = [pattern_maint[x] if x in pattern_maint else x for x in data['maint']]
pattern_dorrs = {'2':2,'3':3,'4':4,'5more':5}
data['doors'] = [pattern_dorrs[x] if x in pattern_dorrs else x for x in data['doors']]
pattern_persons = {'2':2,'4':4,'more':6}
data['persons'] = [pattern_persons[x] if x in pattern_persons else x for x in data['persons']]
pattern_lug_boot = {'small':1,'med':2,'big':3}
data['lug_boot'] = [pattern_lug_boot[x] if x in pattern_lug_boot else x for x in data['lug_boot']]
pattern_safety = {'low':1,'med':2,'high':3}
data['safety'] = [pattern_safety[x] if x in pattern_safety else x for x in data['safety']]
pattern_class = {'unacc':1,'acc':2,'good':3,'vgood':4}
data['Class_Value'] = [pattern_class[x] if x in pattern_class else x for x in data['Class_Value']]

# 切分訓練與測試資料
Y=data['Class_Value']
X = data.drop('Class_Value',axis=1)
X=X.to_numpy()
Y = Y.to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0, train_size=0.8)
clf = SVC(kernel= 'linear',probability=True)
clf.fit(X_train,y_train)
clf.score(X_test,y_test)
print(clf.score(X_test,y_test))


