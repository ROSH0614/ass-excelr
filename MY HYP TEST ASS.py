#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import scipy.stats as stats


# In[2]:


import pandas as pd

import warnings
warnings.filterwarnings("ignore")


# A F&B manager wants to determine whether there is any significant difference in the diameter of the cutlet between two units. A randomly selected sample of cutlets was collected from both units and measured? Analyze the data and draw inferences at 5% significance level. Please state the assumptions and tests that you carried out to check validity of the assumptions.

# In[3]:


hyp = pd.read_csv("C://Users//91948//Downloads//ass 3//Cutlets.csv")
hyp


# In[4]:


len(hyp)


# In[5]:


np.mean(hyp)


# In[6]:


np.std(hyp)


# ##HEREE  5% significance level tells Confidence level is 95% hence alpha value is 0.05
# 
# and by comparing two units 
# we perform TWO-Tail test in this 
# 
# since its two tail test 
# alpha/2 = 0.025

# In[7]:


sns.distplot(hyp)


#  its normally distributed so we can perform the test further

# # we test using paired t test

# In[8]:


from scipy.stats import ttest_rel


# In[9]:


ttest_rel(hyp['Unit A'],hyp['Unit B'],nan_policy='omit')


# # we test using two sample t test

# In[10]:


from scipy.stats import ttest_ind


# In[11]:


_,p_value=ttest_ind(hyp['Unit A'],hyp['Unit B'])


# In[12]:


print(p_value)


# In[13]:


if p_value< 0.05:
    print("we are rejecting null hypothesis")
else:
    p_value> 0.05
    print("we are accepting null hypothesis")


# since there's p_value is more than significance level (5% = 0.05) stated in both the test performed ,  null hypothesis is accepted 

# In[14]:


myhyp= pd.read_csv("C://Users//91948//Downloads//ass 3//LabTAT.csv")


# In[15]:


myhyp


# A hospital wants to determine whether there is any difference in the average Turn Around Time (TAT) of reports of the laboratories on their preferred list. They collected a random sample and recorded TAT for reports of 4 laboratories. TAT is defined as sample collected to report dispatch.
#    
#   Analyze the data and determine whether there is any difference in average TAT among the different laboratories at 5% significance level.
# 

# In[16]:


len(myhyp)


# In[17]:


np.mean(myhyp)


# In[18]:


np.std(myhyp)


# bussiness problem : To determine whether there is any difference in the average Turn Around Time (TAT) of reports of the laboratories on their preferred list.

# In[19]:


sns.distplot(myhyp)


# follows a normal distribution, since all the four laboratories are dependent on one factor time i.e TAT we follow one_way anova test here
# 
# 
# considering the first three samples 

# In[20]:


myhyp.iloc[0:3]


# In[21]:


myhyp.iloc[0:3].mean()


# In[22]:


myhyp.iloc[0:3].std()


# In[23]:


stats.f_oneway(myhyp.iloc[:,0],myhyp.iloc[:,1],myhyp.iloc[:,2],myhyp.iloc[:,3])


# In[24]:


p_value= 2.1156708949992414e-57


# In[25]:


if p_value< 0.05:
    print("we are rejecting null hypothesis")
else:
    p_value> 0.05
    print("we are accepting null hypothesis")


# since the value is p is lower than the given signifance level , accept the null hypothesis in this case 

# Qn 3) Sales of products in four different regions is tabulated for males and females. Find if male-female buyer rations are similar across regions.

# In[26]:


hypt= pd.read_csv("C://Users//91948//Downloads//ass 3//BuyerRatio.csv")


# In[27]:


hypt


# picking up the categorical coloumn 

# In[29]:


hypt.info()


# In[30]:


hypt_1=hypt.drop(columns=['Observed Values'])


# In[31]:


hypt_2=np.array(hypt_1)
hypt_2


# In[32]:


from scipy.stats import chi2_contingency


# In[33]:


chi2_contingency(hypt_2)


# P-value is 0.6603 > 0.05=>P high Ho fly => Accept Ho, hence Average are sameAs per results we can say that there is proportion of male and female buying is similar  As per results we can say that there is proportion of male and female buying is similar

#  Qn 4) TeleCall uses 4 centers around the globe to process customer order forms. They audit a certain %  of the customer order forms. Any error in order form renders it defective and has to be reworked before processing.  The manager wants to check whether the defective %  varies by centre. Please analyze the data at 5% significance level and help the manager draw appropriate inferences

# In[34]:


hypp= pd.read_csv("C://Users//91948//Downloads//ass 3//Costomer+OrderForm.csv")
hypp


# In[35]:


data=hypp.apply(pd.value_counts)
data


# In[36]:


chi2_contingency(data)


# P-value is 0.227 > 0.05=>P high Ho fly => Accept Ho, hence Average are sameP-value is 0.227 > 0.05=>P high Ho fly => Accept Ho, hence Average are sameAs per results we can say that all the canters are equal.As per results we can say that all the centers are equal

# In[ ]:




