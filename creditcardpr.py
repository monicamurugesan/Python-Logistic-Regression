# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 15:20:58 2020

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
cred=pd.read_csv("H://RStudio//assignment//assignments1//Logistic Regression//creditcard.csv")
cred.info()
cred.head()
cred.isnull()
cred['card']=cred['card'].replace({'yes':0,'no':1})
cred['owner']=cred['owner'].replace({'yes':0,'no':1})
cred['selfemp']=cred['selfemp'].replace({'yes':0,'no':1})
x=cred.drop('card',axis=1)
y=cred.card
modcredit=LogisticRegression()
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
len(x_train)
len(x_test)
modcredit.fit(x_train,y_train)
modcredit.score(x_test,y_test)
y_pred=modcredit.predict(x_test)

plt.show()
cm1=confusion_matrix(y_test,y_pred)
sac=accuracy_score(y_test,y_pred)
accper=sac*100
accper

plt.figure(figsize=(10,10))
sns.heatmap(cm1,annot=True)
y_pred_proba = modcredit.predict_proba(x_test)[::,1]
#pyplot.plot(fpr, tpr, linestyle='--', label='No Skill')
fpr, tpr, _ =metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]
vif["features"] = x.columns
vif.round(1)

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
plt.show()
