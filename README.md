# Data_Mining_Project

Kmean_DBSCAN
------------

本研究針對網路提供之線上 資料集 進行 分群的訓練和績效的預測， 隨著大量資料的快速累積以及演算法與雲端運算的發展，大數據分析已經成為學術界與各行業關注與學習的焦點。從巨量資料中篩選有用的資訊，其運用的方法為資料探勘。本研究將使用Python撰寫K-means、階層式分群和DBSCAN之程式進行模型訓練，分別使用 K-means、階層式分群、DBSCAN，將資料分成20群，並比較分群所花費時間，自選評估指標比較分群結果品質，包括使用 Purity指標。K-means所花費的時間約莫 60秒，績效純度為 0.263；階層式分群所花費的時間約莫 108秒，績效純度為 0.204，DBSCAN所花費的時間約莫 40秒，績效純度為 0.09。可以得知K均值可以用於稀疏的高維資料，如文檔資料。DBSCAN通常在這類資料上的性能很差，因為對於高維資料，傳統的歐幾里得密度定義不能很好處理它們。


KNN_DecisionTree
------------
本研究針對台灣客戶的拖欠付款情況，並比較了六種數據挖掘方法中拖欠概率的預測準確性。隨著大量資料的快速累積以及演算法與雲端運算的發展，大數據分析已經成為學術界與各行業關注與學習的焦點。從巨量資料中篩選有用的資訊，其運用的方法為資料探勘。本研究將使用Python撰寫最近鄰居分類器和決策數之程式進行模型訓練，並透過兩個分類器來比較，哪一個分類器在該資料集有 比較好的績效。在KNN和決策樹訓練之下，可以得知KNN在信用卡欠款資料集的分類準確度，已收斂在 0.78 。而決策樹的分類準確度則在 0.825。從績效上的角度來看，可以知道決策樹的分類在這筆資料集上，是比 KNN 來的要好。


TPassenger Hotspot Forecasting 
------------
Based on the Aidea website, this study predicts the hot spots of carrying passengers. With the rapid accumulation for large amounts of data and the development of algorithms with cloud computing, large data analysis has become a focus of academic and administrative attention. The method used to screen useful information from mass data is data mining. In this study, we will use Python to write a random forest program for model training. However, the data provided in this topic are GPS data of taxi time and location in Neihu District. The data covers the period from 2016-02-01 08:00:00 to 2017-01-31 23:59:59 with a total of 4,118,812 entries. The forecasting method of passenger hot spots is as follows: the demand for rides in each period of time (a month) in the Neihu District is predicted. In this study, HIRE_COUNT column value: the predicted number of car demand in the Neihu District during this period, which was empty when downloaded. RMSE was calculated by filling in the predicted value in this column (positive integer value, 0~n) to achieve the difference between the training model and the predicted value.

本研究的目的是利用python建構隨機森林的預測模型，並計算RMSE值來了解預測模型的誤差。RMSE則 是一種常用的測量數值之間差異的量度，其數值常為模型預測的值或是被觀察到的估計值。RMSE代表預測的值和觀察到的值之差的樣本標準差（sample standard deviation），當這些差值是以資料樣本來估計時他們通常被稱為殘差；當這些差值不以樣本來計算時，通常被稱為預測誤差(prediction errors)。RMSE是一個好的準度的量度。 本研究最終透過演算法將會參照剩餘測試資料（60％）的真實數值 (Ground Truth）來驗證與計算分數，最終得分為 10.9339。



SVM_Adaboost
------------
本研究針對汽車交易資料集和網路釣魚資料集進行分類器的訓練和績效的預測，隨著大量資料的快速累積以及演算法與雲端運算的發展，大數據分析已經成為學術界與各行業關注與學習的焦點。從巨量資料中篩選有用的訊，其運用的方法為資料探勘。本研究將使用Python撰寫SVM和Adaboost之程式進行模型訓練並透過兩個分類器來比較，哪一個分類器在該資料集有比較好的績效。在SVM和Adaboost訓練之下，可以得知SVM在汽車交易資料集的分類準確度比網路釣魚資料集的績效來得較高。而Adaboost的分類準確度則在網路釣魚資料集的績效比汽車交易資料集的績效來得較高。從績效上的角度來看可以知道SVM的分類在汽車交易資料集上是比 Adaboost來的要好 Adaboost的分類在網路釣魚資料集的績效比SVM還要好 。
