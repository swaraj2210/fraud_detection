#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression 


# In[2]:


df=pd.read_csv("payment_fraud.csv")


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.info()


# In[6]:


df.describe(percentiles=[0.25,0.5,0.75,0.90,0.95])


# In[7]:


df.isna().sum()


# In[8]:


sns.countplot(df['paymentMethod'])


# In[9]:


sns.countplot(df['label'])


# In[10]:


for f in ['accountAgeDays','numItems','localTime','paymentMethodAgeDays','label']:
    plt.figure(figsize=(10, 10))
    sns.displot(df[f])
    plt.title(f)
 


# In[11]:


feature=['accountAgeDays','numItems','localTime','paymentMethodAgeDays']


# In[12]:


def remove_outliers(df, feature):
    lower_bound = df[feature].mean() - (3 * df[feature].std())
    upper_bound = df[feature].mean() + (3 * df[feature].std())
    df.loc[df[feature] < lower_bound, feature] = lower_bound
    df.loc[df[feature] > upper_bound, feature] = upper_bound



# In[13]:


for f in ['accountAgeDays','numItems','localTime','paymentMethodAgeDays']:
    remove_outliers(df, f)


# In[14]:


for f in ['accountAgeDays','numItems','localTime','paymentMethodAgeDays']:
    plt.figure(figsize=(10, 10))
    sns.displot(df[f])
    plt.title(f)


# In[15]:


from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
df['paymentMethod']=label_encoder.fit_transform(df['paymentMethod'])


# In[16]:


df.head()


# In[17]:


plt.figure(figsize=(10, 10))
sns.heatmap(df.corr(), annot=True);


# In[18]:


## independent and dependent features
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values


# In[19]:


## scaling 

sc = StandardScaler()
X = sc.fit_transform(X)


# In[20]:


## train test split 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# In[21]:


print("X_train shape: ", X_train.shape)
print("X_test shape: ", X_test.shape)
print("y_train shape: ", y_train.shape)
print("y_test shape: ", y_test.shape)


# In[22]:


## logisticRegression Model
lg = LogisticRegression()

## training
lg.fit(X_train, y_train)


# In[23]:


## prediction 
pred = lg.predict(X_test)


# In[24]:


print("----------------------------------------------------Accuracy------------------------------------------------------")
print(accuracy_score(y_test, pred))
print()

print("---------------------------------------------------Classification Report---------------------------------------------")
print(classification_report(y_test, pred))
print()

print("-------------------------------------------------Confustion Metrics----------------------------------------------------")
plt.figure(figsize=(10, 10));
sns.heatmap(confusion_matrix(y_test, pred), annot=True, fmt='g');


# In[ ]:





# In[ ]:




