#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 16:30:50 2019

@author: xuminghai
"""

"""""

import pandas as pd
import sklearn
import os

os.chdir(r'/Users/xuminghai/Desktop/数据挖掘/实验6') 
data = pd.read_csv('SocialCommunity.csv',index_col = 0)

data['Hub.score'] = data['Hub.score'].map(lambda x: 0.6 if x > 0.6 else x)
data['Hub.score'].hist(bins=20)


from sklearn.preprocessing import MinMaxScaler

#以下两种方式相同
scaler = MinMaxScaler()  
data['Hub.score'] = scaler.fit_transform(data[['Hub.score']])

test2 = (data['Hub.score']-data['Hub.score'].min())/(data['Hub.score'].max()- data['Hub.score'].min())



data['Authority.score'] = data['Authority.score'].map(lambda x: 0.3 if x > 0.3 else x)
data['Authority.score'] = scaler.fit_transform(data[['Authority.score']])


data['Good.Bad.Rating'] = data['Good.Bad.Rating'].map(lambda x: 66 if x > 66 else x)
data['Good.Bad.Rating'] = scaler.fit_transform(data[['Good.Bad.Rating']])



data['Authority.score_bin'] = pd.cut(data['Authority.score'],[-0.01,0.4,0.6,1],labels=[0,1,2])
data['Hub.score_bin'] = pd.cut(data['Hub.score'],[-0.01,0.4,0.6,1],labels=[0,1,2])
data['Good.Bad.Rating_bin'] = pd.cut(data['Good.Bad.Rating'],[-0.01,0.4,0.7,1],labels=[0,1,2])

from imblearn.under_sampling import RandomUnderSampler

ros = RandomUnderSampler(random_state=0)
data_new,_ = ros.fit_sample(data,data['Good.Bad.Rating_bin'])

from sklearn.cluster import AgglomerativeClustering 

ac=AgglomerativeClustering(n_clusters=None, distance_threshold=0.9,affinity="euclidean")

label=ac.fit_predict(data_new.iloc[:,:3])


import numpy as np
clu_num = len(np.unique(label))


from sklearn.cluster import KMeans
km= KMeans(n_clusters=7)
data_new['label']=km.fit_predict(data_new.iloc[:,:3])
km.cluster_centers_


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
col = ['b','r','g','c','y','m','k']
ax=plt.subplot(111,projection='3d')
for _,data in data_new.iterrows():
    ax.scatter(data[0],data[1],data[2],color = col[int(data['label'])])

