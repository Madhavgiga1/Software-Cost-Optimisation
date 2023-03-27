#!/usr/bin/env python
# coding: utf-8

# In[5]:


pip install liac-arff


# In[50]:


import arff, numpy as np
import pandas as pd
from scipy.io.arff import loadarff
import urllib.request
from io import StringIO, BytesIO
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.model_selection import KFold,cross_val_score
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV


# In[51]:


cocomo81_url = 'http://promise.site.uottawa.ca/SERepository/datasets/cocomo81.arff'
resp81 = urllib.request.urlopen(cocomo81_url)
cocomo81_data, cocomo81_meta = loadarff(StringIO(resp81.read().decode('utf-8')))
cocomo81_df = pd.DataFrame(cocomo81_data)

# Convert dataframe to numpy array
data = cocomo81_df.values


# In[52]:


data.shape


# In[53]:


X=data[:,0:16]
Y=data[:,16:17]


# In[54]:


Y=Y.reshape(63)


# In[55]:


X.shape


# In[56]:


import seaborn as sns
corr_matrix = cocomo81_df.corr()
plt.figure(figsize=(12,10))

# Plot the correlation matrix
sns.heatmap(corr_matrix, cmap='coolwarm', annot=True, fmt='.2f')

# Show the plot
plt.show()


# # Random Forest

# #  Cocomo n_estimator=100 and max_features= 1 to 16

# In[57]:


X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.30, random_state=42)


# In[58]:


lab_enc = preprocessing.LabelEncoder()
training_scores_encoded = lab_enc.fit_transform(y_train)


# In[59]:


n_feature=7
MMRE_list=[]
MdMRE_list=[]
Pred_list= []
for i in np.arange(1,11):   
    clf= RandomForestRegressor(n_estimators=100,max_features=i)
    clf.fit(X_train, training_scores_encoded)
    Y_pred=clf.predict(X_test)
    diff=np.absolute(y_test-Y_pred)
    MRE=diff/y_test
    MMRE=np.mean(MRE)
    MdMRE=np.median(MRE)
    P=MRE[MRE<.25]
    Pred=(P.size/MRE.size) * 100
    MMRE_list.append(MMRE)
    MdMRE_list.append(MdMRE)
    Pred_list.append(Pred)


# In[60]:


MMRE_list


# In[61]:


MdMRE_list


# In[62]:


import matplotlib.pyplot as plt
import numpy as np

x = np.arange(1,11)

plt.plot(x, MMRE_list)
plt.plot(x, MdMRE_list)

plt.legend(['MMRE','MdMRE'], loc='upper right')

plt.show()


# In[63]:


plt.plot(x, Pred_list)
plt.legend(['Pred'], loc='upper right')
plt.show()


# ## COCOMO n_estimator=100 to 2000 and max_features= 7
# 

# In[64]:


MMRE_list=[]
MdMRE_list=[]
Pred_list= []
for i in np.arange(100,1100,100):   
    clf= RandomForestRegressor(n_estimators=i,max_features=7)
    clf.fit(X_train, training_scores_encoded)
    Y_pred=clf.predict(X_test)
    diff=np.absolute(y_test-Y_pred)
    MRE=diff/y_test
    MMRE=np.mean(MRE)
    MdMRE=np.median(MRE)
    P=MRE[MRE<.25]
    Pred=(P.size/MRE.size) * 100
    MMRE_list.append(MMRE)
    MdMRE_list.append(MdMRE)
    Pred_list.append(Pred)


# In[65]:


MMRE_list


# In[66]:


MdMRE_list


# In[67]:


Pred_list


# In[68]:


x = np.arange(100,1100,100)
plt.plot(x, MMRE_list)
plt.plot(x, MdMRE_list)

plt.legend(['MMRE','MdMRE'], loc='upper right')

plt.show()


# In[69]:


plt.plot(x, Pred_list)
plt.legend(['Pred'], loc='upper right')
plt.show()


# # GRIDSEARCH CV

# In[70]:


params_grd={
 'max_features': [1,2,3,4,5,6,7,8,9,10],
  'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}


# In[71]:


clf= RandomForestRegressor()
grd_search=GridSearchCV(estimator = clf, param_grid = params_grd, 
                          cv = 10, n_jobs = -1, verbose = 2)
grd_search.fit(X_train, training_scores_encoded)
best_grid = grd_search.best_estimator_

Y_pred=best_grid.predict(X_test)
diff=np.absolute(y_test-Y_pred)
MRE=diff/y_test
MMRE=np.mean(MRE)
MdMRE=np.median(MRE)
P=MRE[MRE<.25]
Pred=(P.size/MRE.size) * 100


# In[72]:


print(MMRE)
print(MdMRE)
print(Pred)


# In[73]:


grd_search.best_params_


# # SVR

# In[74]:


svr_params = {'kernel': ('linear','rbf','poly'), 'C':[1.5, 10],'gamma': [1e-7, 1e-4],'epsilon':[0.1,0.2,0.5,0.3]}


# In[79]:


from sklearn import svm
svr= svm.SVR()
svr_random_grd_search=RandomizedSearchCV(estimator = svr, param_distributions = svr_params, n_iter=48, cv = 10, random_state=42, n_jobs = -1)
svr_random_grd_search.fit(X_train, y_train)


# In[81]:


def evaluate_model(actual, predicted):
    diff=np.absolute(actual-predicted)
    MRE=diff/actual
    pred = {}
    for x in np.array([.25, .3, .5]):
        P=MRE[MRE<x]
        pred[x] = (P.size/MRE.size) * 100
    return np.mean(MRE), np.median(MRE), pred


# In[82]:


print(svr_random_grd_search.best_params_)
svr_y_predict = svr_random_grd_search.best_estimator_.predict(X_test)
print(evaluate_model(y_test, svr_y_predict))
print(svr_random_grd_search.best_score_)


# In[85]:


print('MMRE=1.206084001859198')
print('MdMRE=0.6855962159178751')


# # Decision Tree

# In[86]:


clf= DecisionTreeRegressor(max_depth=30, min_samples_split=20)
clf.fit(X_train, training_scores_encoded)
Y_pred=clf.predict(X_test)
diff=np.absolute(y_test-Y_pred)
MRE=diff/y_test
MMRE=np.mean(MRE)
MdMRE=np.median(MRE)
P=MRE[MRE<.25]
Pred=(P.size/MRE.size) * 100


# In[87]:


print(MMRE)
print(MdMRE)
print(Pred)


# # Multiple Linear Regression

# In[118]:


from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LinearRegression
import numpy as np

lr = LinearRegression()

lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)
mmre = np.mean((y_pred - y_test) / y_test)
mdmre = np.median(np.abs((y_pred - y_test) / y_test))

print("MMRE:", mmre)
print("MdMRE:", mdmre)


# In[121]:


mmre_values = [1.206084001859198,0.7643307303049793,0.7569677376178159]
mdmre_values = [0.6855962159178751,0.8158589743589744, 0.8394062078272605]
pred_values=[ 15.789473684210526,5.263157894736842,5.263157894736842]

# Labels for the x-axis (i.e. model names)
model_names = ['SVR', 'Random Forest', 'Decision Tree']

# Plot the MMRE values
plt.bar(model_names, mmre_values, color='b', alpha=0.5, align='center')
plt.ylabel('MMRE')
plt.ylim(0, 1.5) # Set the y-axis limit to 0-0.2
plt.title('Model Effectiveness')



plt.show()


# In[124]:


plt.bar(model_names, pred_values, color='b', alpha=0.5, align='center')
plt.ylabel('PRED')
plt.ylim(0, 30) # Set the y-axis limit to 0-0.2
plt.title('Model Effectiveness')


# In[ ]:




