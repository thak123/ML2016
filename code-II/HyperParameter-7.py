import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import numpy as np
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import StratifiedKFold

import matplotlib.pyplot as py
from sklearn.learning_curve import learning_curve

df= pd.read_csv('wdbc.data.txt',header=None)

X=df.loc[:,2:].values
y=df.loc[:,1].values
le= LabelEncoder()
y=le.fit_transform(y)

print le.transform(['M','B'])

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20, random_state=1)
pipe_lr = Pipeline([('scl',StandardScaler()),('pca',PCA(n_components=2)),('clf',LogisticRegression(penalty='l2',random_state=1,C=0.1))])
pipe_lr.fit(X_train,y_train)
print('Test Accuracy: %.3f' % pipe_lr.score(X_test, y_test))

#K Fold Validation Score
scores=[]
scores=cross_val_score(estimator=pipe_lr,X=X_train,y=y_train,cv=10,n_jobs=1)

print('cv Accuracy scores: %s' %scores)
print ('cv Accuracy: %.3f +/- %.3f' %(np.mean(scores),np.std(scores)))

# Stratified K Fold Validation
kfold=StratifiedKFold(y=y_train,n_folds=10,random_state=1)
scores=[]
for k, (train,test) in enumerate(kfold):
	print y_train[train]
	pipe_lr.fit(X_train[train],y_train[train])
	score=pipe_lr.score(X_train[test],y_train[test])
	scores.append(score)
	print('Fold: %s,Class dist:%s ,Acc: %.3f' %(k+1,np.bincount(y_train[train]),score))
	
print ('cv Accuracy: %.3f +/- %.3f' %(np.mean(scores),np.std(scores)))

#Bias Variance Trade-off vs Accuracy and Learning Curve


