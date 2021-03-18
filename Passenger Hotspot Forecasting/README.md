# Data_Mining_Project

Taxi Passenger Hotspot Forecasting 
------------
Based on the Aidea website, this study predicts the hot spots of carrying passengers. With the rapid accumulation for large amounts of data and the development of algorithms with cloud computing, large data analysis has become a focus of academic and administrative attention. The method used to screen useful information from mass data is data mining. In this study, we will use Python to write a random forest program for model training. However, the data provided in this topic are GPS data of taxi time and location in Neihu District. The data covers the period from 2016-02-01 08:00:00 to 2017-01-31 23:59:59 with a total of 4,118,812 entries. The forecasting method of passenger hot spots is as follows: the demand for rides in each period of time (a month) in the Neihu District is predicted. In this study, HIRE_COUNT column value: the predicted number of car demand in the Neihu District during this period, which was empty when downloaded. RMSE was calculated by filling in the predicted value in this column (positive integer value, 0~n) to achieve the difference between the training model and the predicted value.

本研究的目的是利用python建構隨機森林的預測模型，並計算RMSE值來了解預測模型的誤差。RMSE則 是一種常用的測量數值之間差異的量度，其數值常為模型預測的值或是被觀察到的估計值。RMSE代表預測的值和觀察到的值之差的樣本標準差（sample standard deviation），當這些差值是以資料樣本來估計時他們通常被稱為殘差；當這些差值不以樣本來計算時，通常被稱為預測誤差(prediction errors)。RMSE是一個好的準度的量度。 本研究最終透過演算法將會參照剩餘測試資料（60％）的真實數值 (Ground Truth）來驗證與計算分數，最終得分為 10.9339。




