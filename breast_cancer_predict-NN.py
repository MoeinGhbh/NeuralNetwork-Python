#!/usr/bin/env python
# coding: utf-8

# In[80]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[81]:


#خواندن مجموعه داده
cancer = pd.read_csv('data.csv')
print(cancer.columns)


# ### مجموعه داده شامل 569 رکورد و 32 ویژگی است

# In[82]:


print("مشخصات مجموعه داده: {}".format(cancer.shape))


# In[83]:


#چاپ ده ستون اول داده
cancer.head()


# In[84]:


#مشخصات ستون هدف یعنی تشخیص بیماری
print(cancer.groupby('diagnosis').size())


# In[85]:


#رسم نمودار ستون هدف
import seaborn as sns

sns.countplot(cancer['diagnosis'],label="Count")


# In[86]:


#cancer.info()


# ### تنظیم ستون هدف و ستون های ورودی 

# In[87]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(cancer.loc[:, cancer.columns != 'diagnosis'], cancer['diagnosis'], stratify=cancer['diagnosis'], random_state=66)


# ### استفاده از الگوریتم شبکه عصبی برای ایجاد مدل پیش بینی سرطان سینه

#   به دلیل عدم استفاده از اسکیل کردن داده دقت بسیار پایین است با استفاده از تابعMinMaxScaler  

# In[89]:


#استفاده از تابع برای اسکیل کردن مجموعه داده برای افزایش دقت شبکه عصبی
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(random_state=0)

mlp.fit(X_train_scaled, y_train)
#دقت مدل بعد از پیش پردازش مجموعه داده
print("Accuracy on training set: {:.3f}".format(
    mlp.score(X_train_scaled, y_train)))
print("Accuracy on test set: {:.3f}".format(mlp.score(X_test_scaled, y_test)))


# In[90]:


#دقت مدل بدون پیش پردازش داده
mlp = MLPClassifier(max_iter=1000, alpha=1, random_state=0)
mlp.fit(X_train, y_train)

print("Accuracy on training set: {:.3f}".format(
    mlp.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(mlp.score(X_test, y_test)))


# In[91]:


cancer_features = [x for i,x in enumerate(cancer.columns) if i!=0]
plt.figure(figsize=(20, 5))
plt.imshow(mlp.coefs_[0], interpolation='none', cmap='viridis')
plt.yticks(range(30), cancer_features)
plt.xlabel("Columns in weight matrix")
plt.ylabel("Input feature")
plt.colorbar()


# نمودار فوق وزنه های یادگیری اتصال ورودی به اولین لایه پنهان را نشان می دهد. ردیف های موجود در این طرح با 30 ویژگی ورودی مطابقت دارد ، در حالی که ستون ها با 100 واحد پنهان مطابقت دارند. رنگهای روشن مقدارهای مثبت زیادی را نشان می دهند ، در حالی که رنگهای تیره مقادیر منفی را نشان می دهند.
# 
# یک استنباط احتمالی که می توانیم ایجاد کنیم این است که ویژگی هایی که وزن بسیار کمی برای همه واحدهای پنهان دارند برای مدل "از اهمیت کمتری" برخوردار هستند. می توانیم ببینیم که "میانگین صافی" و "متوسط جمع و جور بودن" ، علاوه بر ویژگی های یافت شده بین "خطای صافی" و "خطای ابعاد فراکتال" ، نسبت به سایر ویژگی ها دارای وزن نسبتاً کمی نیز هستند. این بدان معنی است که این ویژگی ها از اهمیت کمتری برخوردار هستند .

# In[ ]:




