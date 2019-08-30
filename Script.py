# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 07:28:04 2017

@author: u396415
"""

L = ['mydata.txt', 3.14, 10]

print( L[0]) # IN python First element starts with 0
del L[2]  # delete the first element

print(len(L))  # length of L

L.append(-1)  # add -1 at the end of the list

#Forloop 

for e in range(1,10): # last wont be included in the range,and this is how we write range
    print(e)

# or 

for e in range(len(L)):
    print(e)
    
# Then there is array in numpy
# but they are not of much use, since they can only take 1data type at a time
#you cant mix the datatypes

# Plotting in python 

#ctrl +1 to cooment/uncomment the scripts
#ctrl + page up/ down to switch the tab in editor

import numpy as np
import matplotlib.pyplot as plt 

a= range(1,10,2)
b=[2,20,2]

plt.plot(a)
plt.show()

#functions

#functions start with a keyword def
# s is name of the function

def s(t):
    return x*t + y*t
x = 5
y = 4
s(3) 

#Functinons arguments can be passed as default value too

def s(t, a=4):
    return x*t + y*a*t
    
# if else placeholder
If this:
   then 
elif this :
    --
else:
    
    
 #class is nothing but it packs together the variables and function  
    
# Pandas
import pandas as pd

# here I will include 1 sample dataset
import sklearn
from sklearn import datasets

data=pd.read_csv()

#sample
iris=datasets.load_iris()
# since it has many things, just picking the data part
data=iris.data
data=pd.DataFrame(data)
len(data) # will give you number of rows 

#to get the columns names in the list
list(data.columns.values) # for giving name of the columns

#Now to access a columns, you can either yoour index or name of the column

data[1] # If this would be string, then it would have been data["column1"]

# to see many columns 

data[["column1", "column2"]]

data.head()
data.describe() # to see the data structure # almost similit R summary

data.shape
data.dtypes # for finding dtypes
#locations
data.iloc[:,:] # all dataframe
data.iloc[:5,:]# first 5 rows and all the columns
data.iloc[5:,:]#startingfrom 5 rows and all the columns
data.iloc[5:,5:] #from 5th row nd 5th column 
data.iloc[5,:]#only 5th row and all column

#loc to index by labesls

data.loc[:5, 'column1']

# if more columns then take this one

data.loc[:5,['col1', 'col2']]

type(data["col1"]) # to see the datatype of you can use dtype
 

data.corr() # to find the correlation

#to drop any columns axis=1 fro column and axis= 0 for rows
data.drop('id',axis=1,inplace=True)
#for dropping more variables
data=data.drop(labels=['id','area','perimeter'],axis=1)

# for an kind of subseeting in the data 
# that part of data column motoe valus is E and column screw is E 
data[(data.motor=='E') & (data.screw=='E')]
 
 # To change the variabe type in     
     
data2.col1 = pd.to_numeric(data2.col1, errors='coerce')

# Appending or merging data 

data= data1.append(data2, ignore_index=True) # it will be on rows 

# if on rows again 

data=pd.concat(data1, data2)

# if on columns 

data= pd.concat([data1, data2], axis=1)

# merging two data on ids 

data=pd.merge(data1, data2, on='id') #it will be an inner join

data =pd.merge(data1, data2,how='right') #right join

 

# splitting the data in train and test
from sklearn.cross_validation import train_test_split

data_train, data_test, label_train, label_test = train_test_split(X, Y, test_size=0.33, random_state=1)
    
     
# preprocessing and standarddizing the data      
from sklearn.preprocessing import Normalizer

T = preprocessing.Normalizer().fit(data_train)
Train = T.transform(data_train)
     
 # PCA Example     
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(a)
T = pca.transform(a)
 

#Clustering       
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5)
df=pd.read_csv()
kmeans.fit(df)
KMeans(copy_x=True, init='k-means++', max_iter=300, n_clusters=5, n_init=10,
    n_jobs=1, precompute_distances='auto', random_state=None, tol=0.0001,
    verbose=0)

labels = kmeans.predict(df)
centroids = kmeans.cluster_centers_

# KNN

from sklearn.neighbors import KNeighborsClassifier
X_train=pd.DataFrame(Train_pca)

model_knn = KNeighborsClassifier(n_neighbors=1)
model_k=model_knn.fit(X_train, Y_t) 

# You can pass in a dframe or an ndarray
R=model_knn.predict(X_Test)


test_label=label_test.values.ravel()

model_k.score(X_Test,test_label)


#SVM
from sklearn.svm import SVC 

svc=SVC(kernel='linear', C=C)

svc.fit(X_train,y_train)
svc.score(X_test,y_test)






