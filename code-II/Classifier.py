import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Imputer
from sklearn import cross_validation
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.metrics import roc_curve , auc,confusion_matrix
from scipy import interp

from sklearn.pipeline import Pipeline
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import StratifiedKFold
from sklearn.learning_curve import learning_curve
from sklearn.learning_curve import validation_curve

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score, f1_score
#private classes
from SBS import SBS

class Classifier():
	def __init__(self,dataset):
	
		self.df=None
		self.dataset=dataset
		self.X_train=None
		self.X_test=None 
		self.y_train=None
		self.y_test=None
		self.X_train_std=None
		self.X_test_std=None
		self.stdsc=None
		self.lr=None		#one to be used by all the functions
	def clf_fit_transform(self):
		#import dataset
		self.df= pd.read_csv(self.dataset,na_values=["?"])
		
		#clean dataset
		#use median,most_frequent,mean
		imr = Imputer(missing_values='NaN', strategy='mean', axis=0,copy=False)

		imr.fit(self.df)
		X_imputed_df = pd.DataFrame(imr.transform(self.df.values), columns = self.df.columns)


		X_imputed_df.drop(['id'],1,inplace=True)


		X= np.array(X_imputed_df.drop(['class'],1))
		y=np.array(X_imputed_df['class'])

		le= LabelEncoder()
		y=le.fit_transform(y)

	
		#cross validation
		self.X_train, self.X_test ,self.y_train,self.y_test = cross_validation.train_test_split(X,y,test_size=0.2,random_state=0)

		# define the object
		self.stdsc = StandardScaler()

		self.X_train_std= self.stdsc.fit_transform(self.X_train)

		# once it learns it can apply on other inputs
		self.X_test_std= self.stdsc.transform(self.X_test)
			

	def default_accuracy_lr(self):
		lr= LogisticRegression()
		lr.fit(self.X_train_std,self.y_train)
		accuracy=lr.score(self.X_test_std,self.y_test)
		print 'Default Accuracy'
		print(accuracy)
	
	def weight_coefficent_lr(self):
		fig = plt.figure()
		ax = plt.subplot(111)
    
		colors = ['blue', 'green', 'red', 'cyan',           'magenta','yellow', 'black',           'pink', 'lightblue',           'gray', 'indigo', 'orange']

		weights, params = [], []
		for c in np.arange(-4, 6):
			lr = LogisticRegression(penalty='l1', C=10**c, random_state=0)
			lr.fit(self.X_train_std, self.y_train)
			weights.append(lr.coef_[0])
			params.append(10**c)

		weights = np.array(weights)

		for column, color in zip(range(weights.shape[1]), colors):
			plt.plot(params, weights[:, column],
					 label=self.df.columns[column + 1],
					 color=color)
		plt.axhline(0, color='black', linestyle='--', linewidth=3)
		plt.xlim([10**(-5), 10**5])
		plt.ylabel('weight coefficient')
		plt.xlabel('C')
		plt.xscale('log')
		plt.legend(loc='upper left')
		ax.legend(loc='upper center', 
				  bbox_to_anchor=(1.38, 1.03),
				  ncol=1, fancybox=True)
		plt.savefig('./figures/l1_path.png', dpi=300)
		plt.show()
	def l1l2(self):
		# lr = LogisticRegression(penalty='l1',C=0.1)
		
		lr = Pipeline([	
			('scl',StandardScaler()),
			# ('pca',PCA(n_components=2)),
			('clf',LogisticRegression(penalty='l2',C=0.1,random_state=0))])
			

		lr.fit(self.X_train, self.y_train)
		y_pred=lr.predict(self.X_test)
		confmat=confusion_matrix(y_true=self.y_test, y_pred=y_pred)
		print confmat

		print('Precision: %.3f' % precision_score(y_true=self.y_test, y_pred=y_pred))
		print('Recall: %.3f' % recall_score(y_true=self.y_test,y_pred=y_pred))
		print('F1: %.3f' % f1_score(y_true=self.y_test, y_pred=y_pred))
		print('Accuracy : %.3f' % accuracy_score(y_true=self.y_test,y_pred=y_pred))
		
		fig, ax = plt.subplots(figsize=(2.5, 2.5))
		ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
		for i in range(confmat.shape[0]):
			for j in range(confmat.shape[1]):
					ax.text(x=j,y=i,s=confmat[i,j], va='center',ha='center')
		plt.xlabel('predicted label')
		plt.ylabel('true label')
		plt.show()


		
		
		print 'ji'
		# lr = LogisticRegression(penalty='l1')
		# X_train_std=np.delete(self.X_train_std,
		# lr.fit(X_train_std, self.y_train)
		# print('Training accuracy:', lr.score(self.X_train_std, self.y_train))
		# print('Test accuracy:', lr.score(self.X_test_std, self.y_test))
		
		# lr = LogisticRegression(penalty='l1',C=0.1)
		# lr.fit(self.X_train_std, self.y_train)
		# print('Training accuracy:', lr.score(self.X_train_std, self.y_train))
		# print('Test accuracy:', lr.score(self.X_test_std, self.y_test))
		
	
	def SBS_lf(self):
		#Features using SBS

		lr = LogisticRegression(penalty='l1',C=0.1)

		# selecting features
		sbs = SBS(lr, k_features=1)
		sbs.fit(self.X_train_std, self.y_train)

		# plotting performance of feature subsets
		k_feat = [len(k) for k in sbs.subsets_]

		plt.plot(k_feat, sbs.scores_, marker='o')
		plt.ylim([0.7, 1.1])
		plt.ylabel('Accuracy')
		plt.xlabel('Number of features')
		plt.grid()
		plt.tight_layout()
		# plt.savefig('./sbs.png', dpi=300)
		plt.show()

		print sbs.subsets_

		k5 = list(sbs.subsets_[5])

		print(self.df.columns[1:][k5])

		lr.fit(self.X_train_std, self.y_train)
		print('Training accuracy:', lr.score(self.X_train_std, self.y_train))
		print('Test accuracy:', lr.score(self.X_test_std, self.y_test))
		
		k5 = list(sbs.subsets_[6])

		print(self.df.columns[1:][k5])

		lr.fit(self.X_train_std, self.y_train)
		print('Training accuracy:', lr.score(self.X_train_std, self.y_train))
		print('Test accuracy:', lr.score(self.X_test_std, self.y_test))

		# lr.fit(X_train_std[:, k5], self.y_train)
		# print('Training accuracy:', lr.score(X_train_std[:, k5], y_train))
		# print('Test accuracy:', lr.score(X_test_std[:, k5], y_test))


		for iterating_var in sbs.subsets_:
			k_iter=list(iterating_var)
			lr.fit(self.X_train_std[:, k_iter], self.y_train)
			print(self.df.columns[1:][k_iter])
			print('Training accuracy:', lr.score(self.X_train_std[:, k_iter], self.y_train))
			print('Test accuracy:', lr.score(self.X_test_std[:, k_iter], self.y_test))
			print('\n')
			
	def feature_selection_rf(self):
		#Feature Label using Random Forest

		feat_labels = self.df.columns[1:]

		forest = RandomForestClassifier(n_estimators=10000,
										random_state=0,
										n_jobs=-1)

		forest.fit(self.X_train, self.y_train)
		importances = forest.feature_importances_

		indices = np.argsort(importances)[::-1]

		for f in range(self.X_train.shape[1]):
			print("%2d) %-*s %f" % (f + 1, 30, 
									feat_labels[indices[f]], 
									importances[indices[f]]))

		plt.title('Feature Importances')
		plt.bar(range(self.X_train.shape[1]), 
				importances[indices],
				color='lightblue', 
				align='center')

		plt.xticks(range(self.X_train.shape[1]), 
				   feat_labels[indices], rotation=90)
		plt.xlim([-1, self.X_train.shape[1]])
		plt.tight_layout()
		#plt.savefig('./random_forest.png', dpi=300)
		plt.show()
	
	def PCA(self):
		#PCA components/Dimensionality Reduction
		cov_mat = np.cov(self.X_train_std.T)
		eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)

		print('\nEigenvalues \n%s' % eigen_vals)

		tot = sum(eigen_vals)
		var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
		cum_var_exp = np.cumsum(var_exp)

		#plot the cumulative graph

		plt.bar(range(1, 10), var_exp, alpha=0.5, align='center',
				label='individual explained variance')
		plt.step(range(1, 10), cum_var_exp, where='mid',
				 label='cumulative explained variance')
		plt.ylabel('Explained variance ratio')
		plt.xlabel('Principal components')
		plt.legend(loc='best')
		plt.tight_layout()
		# plt.savefig('./figures/pca1.png', dpi=300)
		plt.show()


		# Make a list of (eigenvalue, eigenvector) tuples
		eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i])
					   for i in range(len(eigen_vals))]

		# Sort the (eigenvalue, eigenvector) tuples from high to low
		eigen_pairs.sort(reverse=True)

		w = np.hstack((eigen_pairs[0][1][:, np.newaxis],
					   eigen_pairs[1][1][:, np.newaxis]))
		print('Matrix W:\n', w)

		X_train_pca = self.X_train_std.dot(w)

		colors = ['r', 'b', 'g']
		markers = ['s', 'x', 'o']

		for l, c, m in zip(np.unique(self.y_train), colors, markers):
			plt.scatter(X_train_pca[self.y_train == l, 0], 
						X_train_pca[self.y_train == l, 1], 
						c=c, label=l, marker=m)

		plt.xlabel('PC 1')
		plt.ylabel('PC 2')
		plt.legend(loc='lower left')
		plt.tight_layout()
		# plt.savefig('./figures/pca2.png', dpi=300)
		plt.show()
		#Roc curve



		pca = PCA()
		X_train_pca = pca.fit_transform(self.X_train_std)
		pca.explained_variance_ratio_
		plt.bar(range(1, 10), pca.explained_variance_ratio_, alpha=0.5, align='center')
		plt.step(range(1, 10), np.cumsum(pca.explained_variance_ratio_), where='mid')
		plt.ylabel('Explained variance ratio')
		plt.xlabel('Principal components')
		plt.show()

		pca = PCA(n_components=2)
		
		X_train_pca = pca.fit_transform(self.X_train_std)
		X_test_pca = pca.transform(self.X_test_std)

		plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1])
		plt.xlabel('PC 1')
		plt.ylabel('PC 2')
		plt.show()
		
		lr = LogisticRegression()
		lr = lr.fit(X_train_pca, self.y_train)
		plot_decision_regions(X_train_pca, self.y_train, classifier=lr)
		plt.xlabel('PC 1')
		plt.ylabel('PC 2')
		plt.legend(loc='lower left')
		plt.tight_layout()
		# plt.savefig('./figures/pca3.png', dpi=300)
		plt.show()

		plot_decision_regions(X_test_pca, self.y_test, classifier=lr)
		plt.xlabel('PC 1')
		plt.ylabel('PC 2')
		plt.legend(loc='lower left')
		plt.tight_layout()
		# plt.savefig('./figures/pca4.png', dpi=300)
		plt.show()

		print lr.score(X_test_pca,self.y_test)
	
	def train(self):
		self.lr= LogisticRegression(penalty='l2',C=0.1)
		self.lr.fit(self.X_train_std,self.y_train)
		
	def predict_unseen(self,X):
		print X
		# Get it to the scale of the model
		X=self.stdsc.transform(X)
		print X
		X= np.array(X)
		print X
		# Reshape the model
		X =  X.reshape(1,-1)
		print X
		prediction =self.lr.predict(X)
		
		probability=self.lr.predict_proba(X)
		# y = self.lr.predict(X)[0]
		# proba = np.max(self.lr.predict_proba(X))
		return prediction, probability
		
		
	
#dimensionlity 6 code ends

	def pipe_kf_validation(self):

		pipe_lr = Pipeline([('scl',StandardScaler()),
		# ('pca',PCA(n_components=2)),
		('clf',LogisticRegression(penalty='l1',random_state=1,C=0.1))])
		pipe_lr.fit(self.X_train,self.y_train)
		print('Test Accuracy: %.3f' % pipe_lr.score(self.X_test, self.y_test))

		#K Fold Validation Score
		scores=[]
		scores=cross_val_score(estimator=pipe_lr,X=self.X_train,y=self.y_train,cv=10,n_jobs=1)

		print('cv Accuracy scores: %s' %scores)
		print ('cv Accuracy: %.3f +/- %.3f' %(np.mean(scores),np.std(scores)))

		# Stratified K Fold Validation
		kfold=StratifiedKFold(y=self.y_train,n_folds=10,random_state=1)
		scores=[]
		for k, (train,test) in enumerate(kfold):
			pipe_lr.fit(self.X_train[train],self.y_train[train])
			score=pipe_lr.score(self.X_train[test],self.y_train[test])
			scores.append(score)
			print('Fold: %s,Class dist:%s ,Acc: %.3f' %(k+1,np.bincount(self.y_train[train]),score))
			
		print ('cv Accuracy: %.3f +/- %.3f' %(np.mean(scores),np.std(scores)))

#Bias Variance Trade-off vs Accuracy and Learning Curve


#Hyper Parameter-8 Code
	def clf_learning_curve(self): 
		#commented pca ,if unlocked can reduce the plot curve
		pipe_lr = Pipeline([	
			('scl',StandardScaler()),
			# ('pca',PCA(n_components=2)),
			('clf',LogisticRegression(penalty='l2',random_state=0))])
		train_sizes, train_scores, test_scores= learning_curve(		estimator=pipe_lr,
			 X=self.X_train,
			 y=self.y_train,
			 train_sizes=np.linspace(0.1,1.0,10),
			 cv=10,
			 n_jobs=1)
		train_mean = np.mean(train_scores,axis=1)
		train_std= np.std(train_scores,axis=1)
		test_mean=np.mean(test_scores,axis=1)
		test_std= np.std(test_scores,axis=1)


		#Plot of Accuracy vs Number of Training Samples
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

	def clf_validation_curve(self):
		#Plot of Accuracy vs Regularisation
		param_range=[0.001,0.01,0.1,1.0,10.0,100.0]
		pipe_lr = Pipeline([	
			('scl',StandardScaler()),
			# ('pca',PCA(n_components=2)),
			('clf',LogisticRegression(penalty='l2',random_state=0))])
		train_scores, test_scores= validation_curve(		estimator=pipe_lr,
			 X=self.X_train,
			 y=self.y_train,
			 param_name='clf__C',
			 param_range=param_range,
			 cv=10)

		train_mean = np.mean(train_scores,axis=1)
		train_std= np.std(train_scores,axis=1)
		test_mean=np.mean(test_scores,axis=1)
		test_std= np.std(test_scores,axis=1)

		plt.plot(param_range, train_mean,
			color='blue',
			marker='o',
			markersize=5,label='training accuracy')
		plt.fill_between(param_range,
			train_mean+train_std,
			train_mean-train_std,
			alpha=0.15,color='blue')
		plt.plot(param_range, test_mean,
			color='green',linestyle='--',
			marker='s',
			markersize=5,label='validation accuracy')
		plt.fill_between(param_range,
			test_mean+test_std,
			test_mean-test_std,
			alpha=0.15,color='green')
		plt.grid()
		plt.xscale('log')
		plt.legend(loc='lower right')
		plt.xlabel('Parameter C')
		plt.ylabel('Accuracy')
		plt.ylim([0.8,1.0])
		plt.show()
		#validationcurve-9 code ends
		
	def clf_roc_curve(self):
		#commented pca ,if unlocked can reduce the plot curve
		pipe_lr = Pipeline([	
			('scl',StandardScaler()),
			# ('pca',PCA(n_components=2)),
			('clf',LogisticRegression(penalty='l2',random_state=0))])

			
		
		X_train2= self.X_train
		cv= StratifiedKFold(self.y_train,n_folds=3,random_state=1)
		fig= plt.figure(figsize=(7,5))
		mean_tpr=0.0
		mean_fpr=np.linspace(0,1,100)
		all_tpr=[]
		for i,(train,test) in enumerate(cv):
			probas=pipe_lr.fit(X_train2[train],self.y_train[train]).predict_proba(X_train2[test])
			
			fpr,tpr, thesholds= roc_curve(self.y_train[test],probas[:,1],pos_label=1)
			
			mean_tpr+=interp(mean_fpr,fpr,tpr)
			mean_tpr[0]=0.0
			roc_auc=auc(fpr,tpr)
			plt.plot(fpr,tpr,lw=1,label='ROC fold %d (area = %0.2f)'% (i+1, roc_auc))

		plt.plot([0, 1],
				 [0, 1],
				 linestyle='--',
				 color=(0.6, 0.6, 0.6),
				 label='random guessing')
		mean_tpr /= len(cv)
		mean_tpr[-1] = 1.0
		mean_auc = auc(mean_fpr, mean_tpr)
		plt.plot(mean_fpr, mean_tpr, 'k--',
				 label='mean ROC (area = %0.2f)' % mean_auc, lw=2)
		plt.plot([0, 0, 1],
				 [0, 1, 1],
				 lw=2,
				 linestyle=':',
				 color='black',
				 label='perfect performance')

		plt.xlim([-0.05, 1.05])
		plt.ylim([-0.05, 1.05])
		plt.xlabel('false positive rate')
		plt.ylabel('true positive rate')
		plt.title('Receiver Operator Characteristic')
		plt.legend(loc="lower right")

		plt.tight_layout()
		# plt.savefig('./figures/roc.png', dpi=300)
		plt.show()

		pipe_lr = pipe_lr.fit(X_train2, self.y_train)
		y_pred2 = pipe_lr.predict(self.X_test)

		
		print('ROC AUC: %.3f' % roc_auc_score(y_true=self.y_test, y_score=y_pred2))
		print('Accuracy: %.3f' % accuracy_score(y_true=self.y_test, y_pred=y_pred2))

	
		

			
def plot_decision_regions(X, y, classifier, resolution=0.02):
	# setup marker generator and color map
	markers = ('s', 'x', 'o', '^', 'v')
	colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
	cmap = ListedColormap(colors[:len(np.unique(y))])
	# plot the decision surface
	x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
	x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
	
	xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),np.arange(x2_min, x2_max, resolution))
	Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
	Z = Z.reshape(xx1.shape)
	plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
	plt.xlim(xx1.min(), xx1.max())
	plt.ylim(xx2.min(), xx2.max())

    # plot class samples
	for idx, cl in enumerate(np.unique(y)):
		plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
					alpha=0.8, c=cmap(idx),
					marker=markers[idx], label=cl)
						

