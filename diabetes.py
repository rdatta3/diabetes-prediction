#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[37]:


diabetes=pd.read_csv('diabetes2.csv')
diabetes


# In[38]:


len(diabetes.index)


# In[39]:


sns.countplot(x='Outcome',data=diabetes)


# In[40]:


diabetes['Glucose'].plot.hist()


# In[41]:


diabetes['BloodPressure'].plot.hist()


# In[42]:


diabetes['SkinThickness'].plot.hist()


# In[43]:


diabetes['Insulin'].plot.hist()


# In[44]:


diabetes['BMI'].plot.hist()


# In[45]:


diabetes.info()


# In[46]:


diabetes.isnull().sum()


# In[47]:


sns.heatmap(diabetes.isnull())


# In[48]:


x=diabetes.drop('Outcome',axis=1)
x


# In[49]:


y=diabetes['Outcome']
y


# In[50]:


from sklearn.model_selection import train_test_split


# In[64]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.01,random_state=0)
y_test


# In[65]:


from sklearn.linear_model import LogisticRegression


# In[66]:


result=LogisticRegression()


# In[67]:


result.fit(x_train,y_train)


# In[68]:


x_test


# In[69]:


y_pred=result.predict(x_test)
y_pred


# In[70]:


data=pd.DataFrame({'Actual':y_test,'Predicted':y_pred})
data


# In[71]:


from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
confusion_matrix(y_test,y_pred)


# In[72]:


print(classification_report(y_test,y_pred))


# In[73]:


accuracy_score(y_test,y_pred)


# In[ ]:




