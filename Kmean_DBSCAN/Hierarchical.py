from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import os
from sklearn import cluster
from sklearn import metrics
import collections
from scipy.cluster.hierarchy import linkage,dendrogram
import matplotlib.pyplot as plt
import time

path = "D:/Python/DN_report3/mini_newsgroups"  # 文件目錄
files = os.listdir(path)  # 得到文件夾下的所有文件名稱
X = []
Y = []

for file in files:  # 尋訪文件夾
    newpath = path + "/" + file
    files2 = os.listdir(newpath)
    for sub in files2:
        if not os.path.isdir(sub):  # 判斷是否是文件夾，不是文件夾打開
            f = open(newpath + "/" + sub,encoding="cp1252")  # 打開文件
            iter_f = iter(f)  # 創建迭代
            str = ""
            for line in iter_f:  # 尋訪文件，一行行尋訪，讀取文本
                str = str + line
            Y.append(file)
            X.append(str)  # 每個文件的文本存到list中

# 詞頻向量化
vectorizer = CountVectorizer(min_df=1)
X_counts = vectorizer.fit_transform(X)

# 進行TF-IDF預處理
transformer = TfidfTransformer()
tfidf = transformer.fit_transform(X_counts)

# 計算純度函式
def purity(result, label):
    total_num = len(label)
    cluster_counter = collections.Counter(result)
    original_counter = collections.Counter(label)

    t = []
    for k in cluster_counter:
        p_k = []
        for j in original_counter:
            count = 0
            for i in range(len(result)):
                if result[i] == k and label[i] == j:  # 求交集
                    count += 1
            p_k.append(count)
        temp_t = max(p_k)
        t.append(temp_t)

    return sum(t) / total_num

# Hierarchical
print("Hierarchical")
start_time = time.time()
hclust = cluster.AgglomerativeClustering(linkage = 'ward', affinity = 'euclidean', n_clusters = 20)
hclust.fit(tfidf.toarray())
cluster_label = hclust.labels_
end_time = time.time()
cost_time = end_time - start_time
print("The Hierarchical cost time is : ",format(cost_time,'5.3f')," seconds.")

# 績效評估
silhouette_avg = metrics.silhouette_score(tfidf.toarray(), cluster_label)
print("Silhouette Coefficient : ",silhouette_avg)
print("Purity : ",purity(cluster_label,Y))

# 繪圖
Z = linkage(tfidf.toarray(), method ='ward',metric='euclidean')
p = dendrogram(Z,truncate_mode='lastp',labels=tfidf.toarray())
plt.show()

print("All Finish!")