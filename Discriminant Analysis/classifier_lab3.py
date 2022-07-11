#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd


# In[3]:


data_1a=pd.read_csv("C:/Users/48507/Desktop/MGR_data_science/clasiffiers/lab3/Materials-20220502/Datasets/dataset1a.csv")
data_1b=pd.read_csv("C:/Users/48507/Desktop/MGR_data_science/clasiffiers/lab3/Materials-20220502/Datasets/dataset1b.csv")


# In[4]:


### check the shape of both data frames
data_1a.shape


# In[5]:


data_1b.shape


# In[6]:


data_1a.head


# In[8]:


data_1b.head


# In[9]:


data_1a.dtypes


# In[10]:


data_1b.dtypes


# In[14]:


### define a categorical variable
 # chyba trzeba usunac kolumne unnamed
dfa=pd.DataFrame(data_1a)
dfa["y"].astype('category')
dfa=dfa.drop(dfa.columns[0],axis=1)

dfb=pd.DataFrame(data_1b)
dfb["y"].astype('category')
dfb=dfb.drop(dfb.columns[0],axis=1)


# In[16]:


print(dfb)
print(dfb.dtypes)


# In[11]:


#for col in dfa.select_dtypes(['object','category']):
  #  print(dfa[col].value_counts())


# In[20]:


import seaborn as sns


# In[13]:


sns.pairplot(dfa,hue="y")


# In[14]:


sns.pairplot(dfb,hue="y")


# In[19]:


from matplotlib import pyplot as plt
import numpy as np


# In[16]:


plt.scatter(dfa.x1, dfa.x2, c=dfa.y, alpha=0.5)
plt.title("Scatter plot for dataset 'a'")
plt.xlabel("feature 1 (x1)")
plt.ylabel("feature 2 (x2)")
##plt.legend(["class 0","class 1"])


# In[17]:


plt.scatter(dfb.x1, dfb.x2, c=dfb.y, alpha=0.5)
plt.title("Scatter plot for dataset 'b'")
plt.xlabel("feature 1 (x1)")
plt.ylabel("feature 2 (x2)")
##plt.legend(["class 0","class 1"])


# In[18]:


import plotly.express as px


# In[21]:


ax=sns.boxplot(x="y",y="x1", data=dfa).set_title('Feature distribution x1 in dataset a')


# In[27]:


ax=sns.boxplot(x="y",y="x2", data=dfa).set_title('Feature distribution x2 for dataset a')


# In[28]:


ax=sns.boxplot(x="y",y="x1", data=dfb).set_title('Feature distribution x1 for dataset b')


# In[29]:


ax=sns.boxplot(x="y",y="x2", data=dfb).set_title('Feature distribution x2 for dataset b')


# In[30]:


from sklearn.model_selection import train_test_split


# In[31]:


### Split the dataset 'a' into train and test set:
y_a=dfa.pop("y")
xa_train, xa_test, ya_train, ya_test=train_test_split(dfa,y_a,test_size=0.3)


# In[32]:


### Split the dataset 'b' into train and test set:
y_b=dfb.pop("y")
xb_train, xb_test, yb_train, yb_test=train_test_split(dfb,y_b,test_size=0.3)


# In[45]:


## Linear Dicriminant Analysis for dataset a
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix

#### train model
LDA_a=LinearDiscriminantAnalysis()
LDA_a.fit(xa_train,ya_train)

#### prediction:
predict_LDA_a = LDA_a.predict(xa_test)
#### posterior probabilities:
proba_LDA_a = LDA_a.predict_proba(xa_test)

#### confusion matrix:
TN_a, FP_a, FN_a, TP_a = confusion_matrix(ya_test,predict_LDA_a).ravel()

#### accuracy:
Accuracy_a = (TN_a+TP_a)/(TN_a+FP_a+FN_a+TP_a)*100


# In[36]:


## Confusion matrix LDA(dataset a)
#### kolumny - aktualna klasa, 
#### wiersze - przewidziane wartości

c_matrix1 = np.matrix([[TP_a, FP_a],
                    [FN_a, TN_a]])
c_matrix1=pd.DataFrame(c_matrix1)
c_matrix1.columns=['positive','negative']
c_matrix1.index=['positive','negative']

c_matrix1


# In[46]:


Accuracy_a


# In[47]:


## posterior probabilities of classification
proba_LDA_a


# In[42]:


## Linear Dicriminant Analysis for dataset b

#### train model
LDA=LinearDiscriminantAnalysis()
LDA.fit(xb_train,yb_train)

#### prediction:
predict_LDA=LDA.predict(xb_test)
#### posterior probabilities:
proba_LDA=LDA.predict_proba(xb_test)

#### confusion matrix:
TN_b, FP_b, FN_b, TP_b= confusion_matrix(yb_test,predict_LDA).ravel()
#### accuracy:

Accuracy_b=(TN_b+TP_b)/(TN_b+FP_b+FN_b+TP_b)*100


# In[80]:


## Confusion matrix LDA(dataset b)
#### kolumny - aktualna klasa, 
#### wiersze - przewidziane wartości
c_matrix2 = np.matrix([[TP_b, FP_b],
                    [FN_b, TN_b]])
c_matrix2=pd.DataFrame(c_matrix2)
c_matrix2.columns=['positive','negative']
c_matrix2.index=['positive','negative']


# In[81]:


c_matrix2


# In[53]:


Accuracy_b


# In[43]:


## posterior probabilities of classification
proba_LDA


# In[38]:


## Quadratic Dicriminant Analysis for dataset a
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

#### train model
QDA_a=QuadraticDiscriminantAnalysis()
QDA_a.fit(xa_train,ya_train)

#### prediction:
predict_QDA_a = QDA_a.predict(xa_test)
#### posterior probabilities:
proba_QDA_a = QDA_a.predict_proba(xa_test)

#### confusion matrix:
TN_qa, FP_qa, FN_qa, TP_qa = confusion_matrix(ya_test,predict_QDA_a).ravel()

#### accuracy:
Accuracy_qa = (TN_qa+TP_qa)/(TN_qa+FP_qa+FN_qa+TP_qa)*100


# In[52]:


## Confusion matrix QDA (dataset a)
#### kolumny - aktualna klasa, 
#### wiersze - przewidziane wartości

c_matrix3 = np.matrix([[TP_qa, FP_qa],
                    [FN_qa, TN_qa]])
c_matrix3=pd.DataFrame(c_matrix3)
c_matrix3.columns=['positive','negative']
c_matrix3.index=['positive','negative']

c_matrix3


# In[40]:


Accuracy_qa 


# In[41]:


## posterior probabilities of classification
proba_QDA_a


# In[48]:


## Quadratic Dicriminant Analysis for dataset b

#### train model
QDA_b=QuadraticDiscriminantAnalysis()
QDA_b.fit(xb_train,yb_train)

#### prediction:
predict_QDA_b = QDA_b.predict(xb_test)
#### posterior probabilities:
proba_QDA_b = QDA_b.predict_proba(xb_test)

#### confusion matrix:
TN_qb, FP_qb, FN_qb, TP_qb = confusion_matrix(yb_test,predict_QDA_b).ravel()

#### accuracy:
Accuracy_qb = (TN_qb+TP_qb)/(TN_qb+FP_qb+FN_qb+TP_qb)*100


# In[51]:


## Confusion matrix QDA (dataset b)
#### kolumny - aktualna klasa, 
#### wiersze - przewidziane wartości

c_matrix4 = np.matrix([[TP_qb, FP_qb],
                    [FN_qb, TN_qb]])
c_matrix4=pd.DataFrame(c_matrix4)
c_matrix4.columns=['positive','negative']
c_matrix4.index=['positive','negative']

c_matrix4


# In[50]:


Accuracy_qb


# In[53]:


## posterior probabilities of classification
proba_QDA_b


# In[57]:


from sklearn import svm


# In[59]:


## Support Vector Machines for dataset a
from sklearn import svm

#### train model:
SVM_a = svm.SVC(probability=True)
SVM_a.fit(xa_train, ya_train)

#### prediction:
predict_SVM_a = SVM_a.predict(xa_test)
#### posterior probabilities:
proba_SVM_a = SVM_a.predict_proba(xa_test)

#### confusion matrix:
TN_sa, FP_sa, FN_sa, TP_sa = confusion_matrix(ya_test,predict_SVM_a).ravel()

#### accuracy:
Accuracy_sa = (TN_sa+TP_sa)/(TN_sa+FP_sa+FN_sa+TP_sa)*100


# In[60]:


## Confusion matrix SVM (dataset a)
#### kolumny - aktualna klasa, 
#### wiersze - przewidziane wartości

c_matrix5 = np.matrix([[TP_sa, FP_sa],
                    [FN_sa, TN_sa]])
c_matrix5=pd.DataFrame(c_matrix5)
c_matrix5.columns=['positive','negative']
c_matrix5.index=['positive','negative']

c_matrix5


# In[61]:


Accuracy_sa 


# In[62]:


## posterior probabilities of classification
proba_SVM_a


# In[63]:


## Support Vector Machines for dataset b
from sklearn import svm

#### train model:
SVM_b = svm.SVC(probability=True)
SVM_b.fit(xb_train, yb_train)

#### prediction:
predict_SVM_b = SVM_b.predict(xb_test)
#### posterior probabilities:
proba_SVM_b = SVM_b.predict_proba(xb_test)

#### confusion matrix:
TN_sb, FP_sb, FN_sb, TP_sb = confusion_matrix(yb_test,predict_SVM_b).ravel()

#### accuracy:
Accuracy_sb = (TN_sb+TP_sb)/(TN_sb+FP_sb+FN_sb+TP_sb)*100


# In[65]:


## Confusion matrix SVM (dataset b)
#### kolumny - aktualna klasa, 
#### wiersze - przewidziane wartości

c_matrix6 = np.matrix([[TP_sb, FP_sb],
                    [FN_sb, TN_sb]])
c_matrix6=pd.DataFrame(c_matrix6)
c_matrix6.columns=['positive','negative']
c_matrix6.index=['positive','negative']

c_matrix6


# In[66]:


Accuracy_sb


# In[68]:


proba_SVM_b 

