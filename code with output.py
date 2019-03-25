#!/usr/bin/env python
# coding: utf-8

# In[14]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix

get_ipython().run_line_magic('matplotlib', 'inline')

# I just like to lowercase and remove spaces from my cols to avoid common errors...
knn_df = pd.read_csv("Admission_Predict.csv", names=["id", "gre", "toefl", "u_rating", "sop", "lor", "cgpa", "research", "pred"], header=0, index_col = 0)
knn_df


# In[10]:


import matplotlib.pyplot as plt
import numpy as np
fig, axes = plt.subplots( nrows=4, ncols=2, figsize=(15,40) )


plt.subplots_adjust( wspace=0.20, hspace=0.20, top=0.97 )
plt.suptitle("Graduate admission data", fontsize=20)
axes[0,0].hist(knn_df.gre)
axes[0,0].set_title("GRE score")
axes[0,1].hist(knn_df.toefl)
axes[0,1].set_title("TOEFL score")
axes[1,0].hist(knn_df.u_rating)
axes[1,0].set_title("University rating")
axes[1,1].hist(knn_df.sop)
axes[1,1].set_title("SOR")
axes[2,0].hist(knn_df.lor)
axes[2,0].set_title("LOR")
axes[2,1].hist(knn_df.cgpa)
axes[2,1].set_title("CGPA")
axes[3,0].hist(knn_df.research)
axes[3,0].set_title("Research")
axes[3,1].hist(knn_df.pred)
axes[3,1].set_title("chance of admit")
plt.show()


# In[11]:


# Scale the data.  I used different scalers for criteria and classes 
# It just worked out to get me better results this time around.

# Use the MinMax Scaler and make the percentage True of False at 65% on the pred column
mm_scaler = MinMaxScaler()
knn_df['pred'] = mm_scaler.fit_transform(knn_df[['pred']])
knn_df['pred'] = knn_df['pred'].apply(lambda x: 0 if x <= 0.65 else 1)
# knn_df['pred'].value_counts() # 0=220 1=180


# Use the Standard Scaler on the criteria
s_scaler = StandardScaler()
s_scaler.fit(knn_df.drop('pred',axis=1))
scaled_features = s_scaler.transform(knn_df.drop('pred',axis=1))


X_train, X_test, y_train, y_test = train_test_split(scaled_features,knn_df['pred'], test_size=0.33)


# In[5]:


error = []

# loop through knn predictions with different k values to find the optimal k value

for i in range(1,40):

    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))
    
# Instead of a screen full of numbers lets plot it

plt.figure(figsize=(16,12))
plt.plot(range(1,40),error,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=12)

plt.title('Error Rate vs. Chosen K Value')
plt.xlabel('Tested K Value')
plt.ylabel('Error Rate')


# In[12]:


knn = KNeighborsClassifier(n_neighbors=30)

knn.fit(X_train,y_train)
pred = knn.predict(X_test)

print(f'Confusion Matrix: \n {confusion_matrix(y_test,pred)} \n \n')
print(f'Classification Report: \n {classification_report(y_test,pred)}')


# In[ ]:





# In[ ]:




