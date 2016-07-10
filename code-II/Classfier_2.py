import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Imputer
from sklearn import cross_validation
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import StratifiedKFold
from sklearn.learning_curve import learning_curve
from sklearn.learning_curve import validation_curve

#private classes
from SBS import SBS

# class Classifier():
	# def __init__(self,dataset)
#import dataset
df=pd.read_csv('breast-cancer-wisconsin.data.txt',na_values=["?"])
#clean dataset

#use median,most_frequent,mean
imr = Imputer(missing_values='NaN', strategy='mean', axis=0,copy=False)

imr.fit(df)
X_imputed_df = pd.DataFrame(imr.transform(df.values), columns = df.columns)


X_imputed_df.drop(['id'],1,inplace=True)


X= np.array(X_imputed_df.drop(['class'],1))
y=np.array(X_imputed_df['class'])

le= LabelEncoder()
y=le.fit_transform(y)

#cross validation
X_train, X_test ,y_train,y_test = cross_validation.train_test_split(X,y,test_size=0.2)

# define the object
stdsc = StandardScaler()

X_train_std= stdsc.fit_transform(X_train)

# once it learns it can apply on other inputs
X_test_std= stdsc.transform(X_test)


lr= LogisticRegression()
lr.fit(X_train_std,y_train)


accuracy=lr.score(X_test_std,y_test)
print 'Default Accuracy'
print(accuracy)


fig = plt.figure()
ax = plt.subplot(111)
    
colors = ['blue', 'green', 'red', 'cyan', 
          'magenta', 'yellow', 'black', 
          'pink', 'lightgreen', 'lightblue', 
          'gray', 'indigo', 'orange']

weights, params = [], []
for c in np.arange(-4, 6):
    lr = LogisticRegression(penalty='l1', C=10**c, random_state=0)
    lr.fit(X_train_std, y_train)
    weights.append(lr.coef_[0])
    params.append(10**c)

weights = np.array(weights)

for column, color in zip(range(weights.shape[1]), colors):
    plt.plot(params, weights[:, column],
             label=df.columns[column + 1],
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



#Features using SBS

lr = LogisticRegression(penalty='l1',C=0.1)

# selecting features
sbs = SBS(lr, k_features=1)
sbs.fit(X_train_std, y_train)

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

print(df.columns[1:][k5])

lr.fit(X_train_std, y_train)
print('Training accuracy:', lr.score(X_train_std, y_train))
print('Test accuracy:', lr.score(X_test_std, y_test))


# lr.fit(X_train_std[:, k5], y_train)
# print('Training accuracy:', lr.score(X_train_std[:, k5], y_train))
# print('Test accuracy:', lr.score(X_test_std[:, k5], y_test))


for iterating_var in sbs.subsets_:
	k_iter=list(iterating_var)
	lr.fit(X_train_std[:, k_iter], y_train)
	print(df.columns[1:][k_iter])
	print('Training accuracy:', lr.score(X_train_std[:, k_iter], y_train))
	print('Test accuracy:', lr.score(X_test_std[:, k_iter], y_test))
	

#Feature Label using Random Forest

feat_labels = df.columns[1:]

forest = RandomForestClassifier(n_estimators=10000,
                                random_state=0,
                                n_jobs=-1)

forest.fit(X_train, y_train)
importances = forest.feature_importances_

indices = np.argsort(importances)[::-1]

for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, 
                            feat_labels[indices[f]], 
                            importances[indices[f]]))

plt.title('Feature Importances')
plt.bar(range(X_train.shape[1]), 
        importances[indices],
        color='lightblue', 
        align='center')

plt.xticks(range(X_train.shape[1]), 
           feat_labels[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
#plt.savefig('./random_forest.png', dpi=300)
plt.show()

#PCA components/Dimensionality Reduction
cov_mat = np.cov(X_train_std.T)
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

X_train_pca = X_train_std.dot(w)

colors = ['r', 'b', 'g']
markers = ['s', 'x', 'o']

for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(X_train_pca[y_train == l, 0], 
                X_train_pca[y_train == l, 1], 
                c=c, label=l, marker=m)

plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
# plt.savefig('./figures/pca2.png', dpi=300)
plt.show()
#Roc curve



pca = PCA()
X_train_pca = pca.fit_transform(X_train_std)
pca.explained_variance_ratio_
plt.bar(range(1, 10), pca.explained_variance_ratio_, alpha=0.5, align='center')
plt.step(range(1, 10), np.cumsum(pca.explained_variance_ratio_), where='mid')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.show()

pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)

plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1])
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.show()

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
						


lr = LogisticRegression()
lr = lr.fit(X_train_pca, y_train)
plot_decision_regions(X_train_pca, y_train, classifier=lr)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
# plt.savefig('./figures/pca3.png', dpi=300)
plt.show()

plot_decision_regions(X_test_pca, y_test, classifier=lr)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
# plt.savefig('./figures/pca4.png', dpi=300)
plt.show()

print lr.score(X_test_pca,y_test)

#dimensionlity 6 code ends

pipe_lr = Pipeline([('scl',StandardScaler()),('pca',PCA(n_components=2)),('clf',LogisticRegression(penalty='l2',random_state=1,C=0.1))])
pipe_lr.fit(X_train,y_train)
print('Test Accuracy: %.3f' % pipe_lr.score(X_test, y_test))


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
	pipe_lr.fit(X_train[train],y_train[train])
	score=pipe_lr.score(X_train[test],y_train[test])
	scores.append(score)
	print('Fold: %s,Class dist:%s ,Acc: %.3f' %(k+1,np.bincount(y_train[train]),score))
	
print ('cv Accuracy: %.3f +/- %.3f' %(np.mean(scores),np.std(scores)))

#Bias Variance Trade-off vs Accuracy and Learning Curve


#Hyper Parameter-8 Code

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


#Plot of Accuracy vs Regularisation
param_range=[0.001,0.01,0.1,1.0,10.0,100.0]
train_scores, test_scores= validation_curve(		estimator=pipe_lr,
	 X=X_train,
	 y=y_train,
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