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

import matplotlib.pyplot as plt
from sklearn.learning_curve import validation_curve
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

df= pd.read_csv('wdbc.data.txt',header=None)

X=df.loc[:,2:].values
y=df.loc[:,1].values
le= LabelEncoder()
y=le.fit_transform(y)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20, random_state=1)

#commented pca ,if unlocked can reduce the plot curve
pipe_svc = Pipeline([	
	('scl',StandardScaler()),
	# ('pca',PCA(n_components=2)),
	('clf',SVC(random_state=1))])

	
pipe_svc.fit(X_train,y_train)
y_pred=pipe_svc.predict(X_test)
confmat=confusion_matrix(y_true=y_test, y_pred=y_pred)
print confmat

fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
	for j in range(confmat.shape[1]):
			ax.text(x=j,y=i,s=confmat[i,j], va='center',ha='center')
plt.xlabel('predicted label')
plt.ylabel('true label')
plt.show()

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score, f1_score

print('Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_pred))
print('Recall: %.3f' % recall_score(y_true=y_test,y_pred=y_pred))
print('F1: %.3f' % f1_score(y_true=y_test, y_pred=y_pred))



