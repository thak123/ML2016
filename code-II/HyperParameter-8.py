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
from sklearn.learning_curve import learning_curve

df= pd.read_csv('wdbc.data.txt',header=None)

X=df.loc[:,2:].values
y=df.loc[:,1].values
le= LabelEncoder()
y=le.fit_transform(y)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20, random_state=1)

#commented pca ,if unlocked can reduce the plot curve
pipe_lr = Pipeline([	
	('scl',StandardScaler()),
	# ('pca',PCA(n_components=2)),
	('clf',LogisticRegression(penalty='l2',random_state=0))])
train_sizes, train_scores, test_scores= learning_curve(		estimator=pipe_lr,
	 X=X_train,
	 y=y_train,
	 train_sizes=np.linspace(0.1,1.0,10),
	 cv=10,
	 n_jobs=1)
train_mean = np.mean(train_scores,axis=1)
train_std= np.std(train_scores,axis=1)
test_mean=np.mean(test_scores,axis=1)
test_std= np.std(test_scores,axis=1)

plt.plot(train_sizes, train_mean,
	color='blue',
	marker='o',
	markersize=5,label='training accuracy')
plt.fill_between(train_sizes,
	train_mean+train_std,
	train_mean-train_std,
	alpha=0.15,color='blue')
plt.plot(train_sizes, test_mean,
	color='green',linestyle='--',
	marker='s',
	markersize=5,label='validation accuracy')
plt.fill_between(train_sizes,
	test_mean+test_std,
	test_mean-test_std,
	alpha=0.15,color='green')
plt.grid()

plt.legend(loc='lower right')
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.ylim([0.8,1.0])
plt.show()

