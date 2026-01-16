#!/usr/bin/env python
# coding: utf-8

# In[2]:


#____IMPORTING DEPENDENCIES____#
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor 
from sklearn import metrics


# In[3]:


#___LOADING DATASET___#
excercise = pd.read_csv('exercise.csv')


# In[4]:


#___LOADING DATASET___#
calories = pd.read_csv('calories.csv')


# In[5]:


#___CONCATINATION OF THE DATA___#
calories_data = pd.concat([excercise,calories['Calories']], axis=1)


# In[6]:


#___CHECKING THE FIRST 5 ROW OF THE DATA___#
calories_data.head()


# In[7]:


#___SHAPE OF THE DATA___#
calories_data.shape


# In[8]:


#___CHECKING FOR MISSING VALUE___#
calories_data.isnull().sum()


# In[9]:


#___CHECKING THE DATA INFO___#
calories_data.info()


# In[10]:


#___DESCRIPTIVE STATISTICS___#
calories_data.describe()


# In[11]:


#___ENCODING CATEGORICAL FEATURES___#
calories_data['Gender'] = calories_data['Gender'].map({'male':0, 'female':1})


# In[12]:


calories_data.head()


# In[13]:


calories_data['Gender'].value_counts()


# In[14]:


#___DROPPING FEATURES THAT ARE NOT NEEDED___#
calories_data = calories_data.drop(columns='User_ID', axis=1)


# In[15]:


#___CORRELATION OF THE DATA___#
calories_data.corr()


# In[16]:


#___PLOTTING THE CORRELATION___#
plt.figure(figsize=(8,6))
sns.heatmap(calories_data.corr(), annot=True, cmap='coolwarm', fmt='.2f')


# In[17]:


#___SEPERATING THE TARGET AND THE FEATURES___#
X = calories_data.drop(columns='Calories', axis=1)
Y = calories_data['Calories']


# In[18]:


print(X)


# In[19]:


print(Y)


# In[20]:


#___SPLITTING THE DATA___#
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,random_state=2)


# In[21]:


print(X.shape, X_train.shape, X_test.shape)


# In[22]:


y_train.mean()


# In[23]:


y_train.std()


# In[24]:


model = XGBRegressor(
    max_depth=3,
    reg_alpha=5,
    reg_lambda=5,
    n_estimators=100,
    learning_rate=0.05
)


# In[25]:


#___TRAINING MODEL___#
model.fit(X_train, y_train)


# In[26]:


#___EVALUATION METRICS OF THE TRAIN DATA___#
train_prediction = model.predict(X_train)
score_1 = metrics.mean_absolute_error(y_train, train_prediction)
print(score_1)


# In[27]:


#___EVALUATION METRICS OF THE TEST DATA___#
test_prediction = model.predict(X_test)
score_2 = metrics.mean_absolute_error(y_test, test_prediction)
print(score_2)


# In[28]:


#___TESTING MODEL WITH DATA___#
input_data = (1,20,166.0,60.0,14.0,94.0,40.3)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
prediction = model.predict(input_data_reshaped)
print(prediction)


# ## SAVING MODEL 

# In[29]:


import pickle 


# In[30]:


filename = 'mymodel.pkl'


# In[31]:


pickle.dump(model, open(filename, 'wb'))


# In[32]:


load_model = pickle.load(open('mymodel.pkl', 'rb'))


# In[33]:


input_data = (1,20,166.0,60.0,14.0,94.0,40.3)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
prediction = load_model.predict(input_data_reshaped)
print(prediction)

