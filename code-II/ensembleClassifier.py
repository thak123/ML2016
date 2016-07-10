import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import cross_validation, linear_model
import numpy as np
from sklearn.preprocessing import Imputer
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from MajorityVoteClassifier import MajorityVoteClassifier
df= pd.read_csv('breast-cancer-wisconsin.data.txt',na_values=["?"])

#use median,most_frequent,mean
imr = Imputer(missing_values='NaN', strategy='mean', axis=0,copy=False)

imr.fit(df)
X_imputed_df = pd.DataFrame(imr.transform(df.values), columns = df.columns)


X_imputed_df.drop(['id'],1,inplace=True)


X= np.array(X_imputed_df.drop(['class'],1))
y=np.array(X_imputed_df['class'])

le= LabelEncoder()
y=le.fit_transform(y)

X_train, X_test ,y_train,y_test = cross_validation.train_test_split(X,y,test_size=0.5)

# define the object
stdsc = StandardScaler()

X_train_std= stdsc.fit_transform(X_train)

# once it learns it can apply on other inputs
X_test_std= stdsc.transform(X_test)


clf1 =LogisticRegression(penalty='l2',C=0.1,random_state=0)
clf2= DecisionTreeClassifier(max_depth=1,criterion='entropy', random_state=0)
clf3= KNeighborsClassifier(n_neighbors=1,p=2,metric='minkowski')


pipe1= Pipeline([['sc',StandardScaler()],['clf',clf1]])
pipe3= Pipeline([['sc',StandardScaler()],['clf',clf3]])

clf_labels=['Logistic Regression','Decision Tree','KNN']

print ('10-fold cross validation:\n')

for clf, label in zip([pipe1,clf2,pipe3], clf_labels):
	scores=cross_val_score(estimator=clf,X=X_train,y=y_train,cv=10,scoring='roc_auc')
	print("Roc AUC :%.3f (+/-%.3f)[%s]" %(scores.mean(),scores.std(),label))

	
mv_clf = MajorityVoteClassifier(classifiers=[pipe1, clf2, pipe3])

clf_labels += ['Majority Voting']
all_clf = [pipe1, clf2, pipe3, mv_clf]

for clf, label in zip(all_clf, clf_labels):
    scores = cross_val_score(estimator=clf,
                             X=X_train,
                             y=y_train,
                             cv=10,
                             scoring='roc_auc')
    print("ROC AUC: %0.2f (+/- %0.2f) [%s]"
          % (scores.mean(), scores.std(), label))


from sklearn.metrics import roc_curve
from sklearn.metrics import auc

colors = ['black', 'orange', 'blue', 'green']
linestyles = [':', '--', '-.', '-']
for clf, label, clr, ls \
        in zip(all_clf,
               clf_labels, colors, linestyles):

    # assuming the label of the positive class is 1
    y_pred = clf.fit(X_train,
                     y_train).predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_true=y_test,
                                     y_score=y_pred)
    roc_auc = auc(x=fpr, y=tpr)
    plt.plot(fpr, tpr,
             color=clr,
             linestyle=ls,
             label='%s (auc = %0.2f)' % (label, roc_auc))

plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1],
         linestyle='--',
         color='gray',
         linewidth=2)

plt.xlim([-0.1, 1.1])
plt.ylim([-0.1, 1.1])
plt.grid()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

# plt.tight_layout()
# plt.savefig('./figures/roc.png', dpi=300)
plt.show()


#Individual Plotting
#Code Ends
print mv_clf.get_params()

from sklearn.grid_search import GridSearchCV

params = {'decisiontreeclassifier__max_depth': [1, 2],
          'pipeline-1__clf__C': [0.001, 0.1, 100.0]}

grid = GridSearchCV(estimator=mv_clf,
                    param_grid=params,
                    cv=10,
                    scoring='roc_auc')
grid.fit(X_train, y_train)

for params, mean_score, scores in grid.grid_scores_:
    print("%0.3f+/-%0.2f %r"
          % (mean_score, scores.std() / 2.0, params))
			  
print('Best parameters: %s' % grid.best_params_)
print('Accuracy: %.2f' % grid.best_score_)

print grid.best_estimator_.classifiers
mv_clf = grid.best_estimator_
mv_clf.set_params(**grid.best_estimator_.get_params())

print mv_clf