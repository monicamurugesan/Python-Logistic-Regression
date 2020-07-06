# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 13:51:17 2020

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
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.formula.api import logit
from scipy import stats
banks=pd.read_excel("H://RStudio//assignment//assignments1//Logistic Regression//bank-full.xls")
banks.head()
banks.info()
banks['job']=banks['job'].replace({'admin.':1,'technician':2,'services':3,'management':4,'retired':5,'blue-collar':6,'entrepreneur':7,'unknown':8,'self-employed':9,'housemaid':10,'student':11,'unemployed':12})
banks['education']=banks['education'].replace({'primary':1,'secondary':2,'tertiary':3,'unknown':4})
banks['marital']=banks['marital'].replace({'single':0,'married':1,'divorced':2})
banks['default']=banks['default'].replace({'yes':0,'no':1})
banks['housing']=banks['housing'].replace({'yes':0,'no':1})
banks['contact']=banks['contact'].replace({'telephone':1,'cellular':2,'unknown':3})
banks['loan']=banks['loan'].replace({'yes':0,'no':1})
banks['poutcome']=banks['poutcome'].replace({'success':1,'failure':0,'unknown':3,'other':6})
banks['month']=banks['month'].replace({'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12})
banks['y']=banks['y'].replace({'yes':0,'no':1})
x=banks.drop('y',axis=1)
y=banks.y
x
sns.lmplot(x="education",y="y",data=banks,logistic=True,y_jitter=0.03,ci=69)
banks.describe()
modelbin=sm.GLM(x['housing'],y,family=sm.families.Binomial())
res=modelbin.fit()
res.summary()
print(np.exp(res.params))
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
len(x_train)
len(x_test)


y_pred=res.predict(x_test['job'])
cut=0.05
pred=np.where(y_pred>cut,1,0)
y_act=y_test


cm1=confusion_matrix(y_act,pred)
sac=accuracy_score(y_act,pred)
accper=sac*100
accper

plt.figure(figsize=(10,10))
sns.heatmap(cm1,annot=True)



#pyplot.plot(fpr, tpr, linestyle='--', label='No Skill')
fpr, tpr, _ =metrics.roc_curve(y_act,  pred)
auc = metrics.roc_auc_score(y_act, pred)

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
print(classification_report(y_act,pred))
plt.show()

