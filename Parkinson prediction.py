#!/usr/bin/env python
# coding: utf-8

# # All Classifier with heat maps
# # Parkinson's Disease prediction
# This dataset contains  biomedical voice measurements from 31 people, 23 with Parkinson disease
# in total 195 rows and 24 columns where the first column is about name, last is status if person is healthy or having disease 
# in between columns are particular voice measures.


# In[1]:


#importing the libraries
#!pip install lux
import numpy as np
import pandas as pd
# import warnings  warnings.filterwarnings('ignore')
import os, sys #system related operations
import lux  # convert the panda into beautiful visulaization
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[2]:


#LOAD THE DATASET
data_set = pd.read_csv('parkinsons-Copy1.csv')
data_set.head(n=15)


# In[3]:


data_set.describe()


# In[4]:


data_set.info()


# In[5]:


data_set.isnull().sum()


# In[6]:


#finding unique values in a column
for i in data_set.columns:
  print("**************************************************",i,"***************************************")
  print()
  print(set(data_set[i].tolist()))
  print()


# In[7]:


#check label imbalance
import matplotlib.pyplot as plt
import seaborn as sns
temp= data_set['status'].value_counts()
temp_df=  pd.DataFrame({'status':temp.index,"values": temp.values})
print(sns.barplot(x="status",y="values",data=temp_df))


# In[8]:


sns.pairplot(data_set)


# In[9]:


def distplot(col):
  sns.distplot(data_set[col])
  plt.show()
for i in list(data_set.columns)[1:]:
  distplot(i)


# In[10]:


#Geting idea of outliers in data
def boxplots(col):
  sns.boxplot(data_set[col])
  plt.show()
for i in list (data_set.select_dtypes(exclude=["object"]).columns)[1:]:
  boxplots(i)


# In[11]:


#finding correlation btwn the data (independent variable should be highly corelated with dependent variable )
plt.figure(figsize=(20,20))
corr= data_set.corr()
sns.heatmap(corr,annot=True)

#now depending on the colour of heat map the corelation can be seen nd highly corelated data is eliminated


# In[12]:


#lets make some final changes in data set 
#sepearting dependent variable and independent variable and droping the ID column
x= data_set.drop(['status','name'],axis=1)
y = data_set['status']


# In[13]:


#lets detect the label balance
#!pip install imblearn
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

print(Counter(y))


# In[14]:


#LETS balance the labels
ros = RandomOverSampler()
x_ros, y_ros= ros.fit_resample(x,y)
print(Counter(y_ros))


# In[15]:


#initialize a minmaxscaler and scale the feature to btw -1to 1 to normal size them
#The MinMazScaler transforms features by scaling them to given range 
#The fit_transform() method fits to the data and then transform it. we don't need to scale the label
#Scale the feature to btwn -1 to 1

#Scaling is important in SVM , KNN where distance btwn two data point is imoprtant

scaler =MinMaxScaler((-1,1))
x= scaler.fit_transform(x_ros)
y = y_ros


# In[16]:


#Applying feature engineering

#applying PCA 


from sklearn.decomposition import PCA
pca= PCA(.95)
x_pca = pca.fit_transform(x)
print(x.shape)
print(x_pca.shape)


# In[17]:


x_train, x_test, y_train, y_test= train_test_split(x_pca,y,test_size=0.2,random_state=7)


# In[18]:


from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score,f1_score

list_met =[]
list_acuracy =[]

#applying all the algorithm

#Logistic Regression

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(C=0.4,max_iter=1000,solver='liblinear')
lr=classifier.fit(x_train,y_train)
y_pred= classifier.predict(x_test)
accuracy_lr= accuracy_score(y_test,y_pred)


#Decision Tree
from sklearn.tree import DecisionTreeClassifier

classifier2= DecisionTreeClassifier(random_state=14)
dt= classifier2.fit(x_train,y_train)
y_pred2= classifier2.predict(x_test)
accuracy_dt= accuracy_score(y_test,y_pred2)

#Random Forest criteria information gain
from sklearn.ensemble import RandomForestClassifier
classifier3= RandomForestClassifier(random_state= 14)
rf= classifier3.fit(x_train,y_train)
y_pred3= classifier3.predict(x_test)
accuracy_rf= accuracy_score(y_test,y_pred3)

#Random Forest criteria entropy
from sklearn.ensemble import RandomForestClassifier
classifier4= RandomForestClassifier(criterion='entropy')
rfe= classifier4.fit(x_train,y_train)
y_pred4= classifier4.predict(x_test)
accuracy_rfe= accuracy_score(y_test,y_pred4)

#SVM(support vector machine)
from sklearn.svm import SVC
model_svm= SVC(cache_size= 100)
svm = model_svm.fit(x_train,y_train)
y_pred5= model_svm.predict(x_test)
accuracy_svm= accuracy_score(y_test,y_pred5)

#KNN

from sklearn.neighbors import KNeighborsClassifier
model_knn3= KNeighborsClassifier(n_neighbors=3)
knn = model_knn3.fit(x_train,y_train)
pred_knn3= model_knn3.predict(x_test)
accuracy_knn= accuracy_score(y_test,pred_knn3)

#Gaussian Naive Bayes

from sklearn.naive_bayes import GaussianNB
gnb= GaussianNB()
gnb= gnb.fit(x_train,y_train)
pred_gnb= gnb.predict(x_test)
accuracy_gnb= accuracy_score(y_test,pred_gnb)

#Bernaulli Naive Bayes

from sklearn.naive_bayes import BernoulliNB
model= BernoulliNB()
bnb= model.fit(x_train,y_train)
pred_bnb= bnb.predict(x_test)
accuracy_bnb= accuracy_score(y_test,pred_bnb)

#Combining ALL the classifier
from sklearn.ensemble import VotingClassifier
evc= VotingClassifier(estimators=[('lr',lr),('rf',rf),('rfe',rfe),('dt',dt),('svm',svm),('knn',knn),('gnb',gnb),('bnb',bnb),],voting='hard',flatten_transform=True)


model_evc= evc.fit(x_train,y_train)
pred_evc= evc.predict(x_test)
accuracy_evc= accuracy_score(y_test,pred_evc)

list1=['Logistic Regression','Decision Tree','RandomForest(information gain)', 'Random Forest(Entropy)','KNN','SVM','gnb','bnb','Voting Classifier']
list2= [accuracy_lr,accuracy_dt,accuracy_rf,accuracy_rfe,accuracy_knn,accuracy_svm,accuracy_gnb,accuracy_bnb,accuracy_evc]
list3=[classifier,classifier2,classifier3,classifier4,model_knn3,model_svm,gnb,model]

df_Accuracy= pd.DataFrame({'Method Used': list1,'Accuracy Score': list2})
print(df_Accuracy)

chart= sns.barplot(x='Method Used', y='Accuracy Score', data= df_Accuracy)
chart.set_xticklabels(chart.get_xticklabels(),rotation=90)
print(chart)


# In[19]:


#THIS classifies extreme gradient boosting - using gradient boosting algorithm for modern data science problem
#It falls under the category of ensemble machine learning
#train the model

model_xg= XGBClassifier()
model_xg.fit(x_train,y_train)


# In[20]:


y_pred = model_xg.predict(x_test)
print(accuracy_score(y_test, y_pred)*100)


# In[21]:


#confusion matrix

from sklearn.metrics import confusion_matrix
cm= confusion_matrix(y_test,model_xg.predict(x_test))


# In[22]:


from sklearn.metrics import f1_score
f1_score(y_test,model_xg.predict(x_test), average='binary')
print("Confusion matrix:\n",cm)


# In[23]:


from sklearn.metrics import roc_curve, confusion_matrix,classification_report,auc
print(classification_report(y_test,model_xg.predict(x_test)))


# In[24]:


for i in list3:
  print("******************************************",i,"****************************************")
  print(classification_report(y_test,i.predict(x_test)))
  print("Confusion matrix:")
  print(confusion_matrix(y_test,i.predict(x_test)))


# Our prediction says the data is best classified by KNeighbour Classifier





