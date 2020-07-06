# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 19:34:19 2020

@author: Hp
"""


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap
from sklearn import metrics
df=pd.read_excel("H://RStudio//assignment//assignments1//Logistic Regression//bank-full.xls")
print(df.head(5))
df.isnull()

df['job']=df['job'].replace({'admin.':1,'technician':2,'services':3,'management':4,'retired':5,'blue-collar':6,'entrepreneur':7,'unknown':8,'self-employed':9,'housemaid':10,'student':11,'unemployed':12})
df['education']=df['education'].replace({'primary':1,'secondary':2,'tertiary':3,'unknown':4})
df['marital']=df['marital'].replace({'single':0,'married':1,'divorced':2})
df['default']=df['default'].replace({'yes':0,'no':1})
df['housing']=df['housing'].replace({'yes':0,'no':1})
df['contact']=df['contact'].replace({'telephone':1,'cellular':2,'unknown':3})
df['loan']=df['loan'].replace({'yes':0,'no':1})
df['poutcome']=df['poutcome'].replace({'success':1,'failure':0,'unknown':3,'other':6})
df['month']=df['month'].replace({'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12})
df['y']=df['y'].replace({'yes':0,'no':1})
x=df.drop('y',axis=1)
y=df.y
model1=LogisticRegression()
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
len(x_train)
len(x_test)
model1.fit(x_train,y_train)
model1.score(x_test,y_test)
y_pred=model1.predict(x_test)

plt.show()
cm1=confusion_matrix(y_test,y_pred)
sac=accuracy_score(y_test,y_pred)
accper=sac*100
accper

plt.figure(figsize=(10,10))
sns.heatmap(cm1,annot=True)
model1.summary()

y_pred_proba = model1.predict_proba(x_test)[::,1]
#pyplot.plot(fpr, tpr, linestyle='--', label='No Skill')
fpr, tpr, _ =metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)

plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.legend(loc=4)
print(classification_report(y_test,y_pred))


