#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Importing required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# In[ ]:


#For ignoring warnings
import warnings
warnings.filterwarnings(action='ignore')


# # Data Pre-processing

# In[ ]:


#Reading the dataset
df=pd.read_csv('forestfires.csv')


# In[ ]:


#Function for pre-processing data according to regression or classification
def preprocessing(df, task="regression"):
    df=pd.read_csv('forestfires.csv')
    
    #Converting String values of month and day into Integer
    month = {'jan':1, 'feb':2, 'mar':3, 'apr':4, 'may':5, 'jun':6, 'jul':7, 'aug':8, 'sep':9, 'oct':10, 'nov':11, 'dec':12,}
    df['month'] = df['month'].map(month)
    day = {'sun':1, 'mon':2, 'tue':3, 'wed':4, 'thu':5, 'fri':6, 'sat':7,}
    df['day'] = df['day'].map(day)
    
    #Converting target feature according to training model used
    if(task=="regression"):

        #Using Log Transformation to reduce skewness of the target feature
        df['area']=np.log(df['area']+1)

        X = df.drop(["area"], axis=1)
        y = df['area']

    #If model is classification, then converting area burnt into 1(if it is greater than 0) else 0
    elif(task=="classification"):
        
        X = df.drop("area", axis=1)
        y = df['area'].apply(lambda x:1 if x>0 else 0)

    #If model is clustering, then storing values of two most important input features(DMC and temp)
    elif(task=="clustering"):
        X = df.iloc[1:500, [5,8]].values
        y= None

    else:
        raise Exception("Enter regression, classification or clustering")

    #Scaling the input features
    scaler=StandardScaler()
    scaler.fit(X)
    if(task=="regression" or task=="classification"):
        X=pd.DataFrame(scaler.transform(X), columns=X.columns)

    return X, y


# # Training the model

# In[ ]:


#Defining model training functions for regression and classification 
def supervised_method(X,y, task="regression"):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True, random_state=1)
    if (task=="regression"):
        model=LinearRegression()
        model.fit(X_train,y_train)
    elif (task=="classification"):
        model=LogisticRegression()
        model.fit(X_train,y_train)
    else:
        raise Exception("Enter regression or classification")
    return model


# In[70]:


#Defining model training functions for clustering 
def unsupervised_method(X):
    model = KMeans(n_clusters=4, init='random', random_state=0)
    y_km = model.fit_predict(X)
    plt.scatter(X[y_km==0,0],X[y_km==0,1],s=20,c='green',marker='x',label='Very low Fire')
    plt.scatter(X[y_km==1,0],X[y_km==1,1],s=20,c='orange',marker='x',label='Low Fire')
    plt.scatter(X[y_km==2,0],X[y_km==2,1],s=20,c='blue',marker='x',label='Medium Fire')
    plt.scatter(X[y_km==3,0],X[y_km==3,1],s=20,c='black',marker='x',label='High Fire')
    plt.scatter(model.cluster_centers_[:,0],model.cluster_centers_[:,1],s=250,marker='*',c='red',label='centroids')
    plt.legend()
    plt.xlabel('DMC')
    plt.ylabel('Temp')
    plt.grid()
    plt.show()
    return model






