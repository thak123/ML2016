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

from sklearn.neighbors import KNeighborsClassifier

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
param_range=[0.001,0.01,0.1,1.0,10.0,100.0,1000.0]

param_grid= [{'clf__C':param_range,
			'clf__kernel':['linear']
			},
			{'clf__C':param_range,
			'clf__gamma':param_range,
			'clf__kernel':['rbf']}]
			
# gs= GridSearchCV(estimator=pipe_svc,
					# param_grid=param_grid,
					# scoring='accuracy',
					# cv=10,
					# n_jobs=1)
# gs =gs.fit(X_train,y_train)
# print(gs.best_score_)
# print(gs.best_params_)

# clf = gs.best_estimator_	
# clf.fit(X_train, y_train)
# print('Test accuracy: %.3f' % clf.score(X_test, y_test))

# scores= cross_val_score(gs,X,y,scoring='accuracy',cv=10)
# print('SVM CV accuracy : %.3f +/-%.3f'%(np.mean(scores),np.std(scores)))


gs = GridSearchCV(
	estimator=DecisionTreeClassifier(random_state=0),
	param_grid=[
	{'max_depth': [1, 2, 3, 4, 5, 6, 7, None]}],
	scoring='accuracy',
	cv=10,n_jobs=1)
	
scores = cross_val_score(gs,	X_train,	y_train,	scoring='accuracy',	cv=5)

print('DecisionTreeClassifier CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))


gs =gs.fit(X_train,y_train)
print(gs.best_score_)
print(gs.best_params_)



pipe_lr = Pipeline([	
	('scl',StandardScaler()),
	 ('pca',PCA(n_components=2)),
	('clf',LogisticRegression(penalty='l2',random_state=0))])
gs= GridSearchCV(estimator=pipe_lr,
					param_grid=[{'clf__C':param_range}],
					scoring='accuracy',
					cv=10,
					n_jobs=1)


gs =gs.fit(X_train,y_train)
print(gs.best_score_)
print(gs.best_params_)



pipe_neighbors=Pipeline([('scl',StandardScaler()),('clf',KNeighborsClassifier())])
metrics       = ['minkowski','euclidean','manhattan'] 
weights       = ['uniform','distance'] #10.0**np.arange(-5,4)
numNeighbors  = np.arange(5,10)
param_grid    = dict(metric_params=metrics,weights=weights,n_neighbors=numNeighbors)
neighbors_range = range(1, 21)
# param_grid = dict(kneighborsclassifier__n_neighbors=neighbors_range)
grid= GridSearchCV(estimator=pipe_neighbors,
					param_grid=param_grid,
					scoring='accuracy',
					cv=10,
					n_jobs=1)
print grid.get_params().keys()		
# pipe = Pipeline.make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=3))
# cross_val_score(pipe, X, y, cv=5, scoring='accuracy').mean()

# using GridSearchCV with Pipeline
# neighbors_range = range(1, 21)
# param_grid = [{'kneighborsclassifier__n_neighbors':neighbors_range}]
# grid = GridSearchCV(pipe_neighbors, param_grid, cv=5, scoring='accuracy')

grid =grid.fit(X_train,y_train)
print(grid.best_score_)
print(grid.best_params_)
