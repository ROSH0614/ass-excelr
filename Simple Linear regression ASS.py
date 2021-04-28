#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


slr= pd.read_csv("C://Users//91948//Downloads//ass 4//delivery_time.csv")
slr


# In[3]:


slr.head()


# In[4]:


slr.describe()


# In[5]:


sns.scatterplot(x=slr['Delivery Time'],y=slr['Sorting Time'])


# In[6]:


sns.distplot(slr['Sorting Time'])


# finding correlation between the lines of the data

# In[7]:


slr.corr()


# In[8]:


X= slr.iloc[:,:-1].values


# In[9]:


X


# In[10]:


Y= slr.iloc[:,1].values


# In[11]:


Y


# after dividing the data we do spliting according to train and test data

# In[13]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)


# In[14]:


from sklearn.linear_model import LinearRegression
simplelinear_regression=LinearRegression()
simplelinear_regression.fit(X_train,Y_train)


# In[17]:


y_predict_test= simplelinear_regression.predict(X_test)


# In[27]:


y_predict_train= simplelinear_regression.predict(X_train)


# predicted y value 

# In[18]:


y_predict_test


# In[28]:


y_predict_train


# Creating the best fit for the delivery (x) vs sorting (y)

# In[26]:


plt.scatter(X_train,Y_train,color='red')
plt.plot(X_train,simplelinear_regression.predict(X_train))
plt.xlabel("Delivery time")
plt.ylabel("Sorting time")
plt.show()


# qn 2   Build a prediction model for Salary_hike

# In[29]:


df= pd.read_csv("C://Users//91948//Downloads//ass 4//Salary_Data.csv")
df


# In[30]:


A= df.iloc[:,:-1]
B= df.iloc[:,1]


# In[95]:


from sklearn.model_selection import train_test_split
A_train,A_test,B_train,B_test= train_test_split(A,B,test_size=1/3,random_state=0)


# In[96]:


from sklearn.linear_model import LinearRegression
simplelinear_regression=LinearRegression()
simplelinear_regression.fit(A_train,B_train)


# In[97]:


B_predict_test= simplelinear_regression.predict(A_test)
B_predict_test


# In[98]:


B_predict_train= simplelinear_regression.predict(A_train)
B_predict_train


# In[99]:


plt.scatter(A_train,B_train,color='red')
plt.plot(X_train,simplelinear_regression.predict(X_train))
plt.xlabel("EXPERIENCE")
plt.ylabel("SALARY")
plt.show()


# same procedure is followed as of the first one 
# 
# since theres no requirement in the EDA here direct linear regression model is been done
# step 1: loading the data
# stepw 2: assign the data or split the data 
# step 3: based on the train and test data is been assigned 
# step 4: implementation of classifiers and regressions 
# step 5: plot the same with best fit line visible
# 
# 

# In[ ]:




