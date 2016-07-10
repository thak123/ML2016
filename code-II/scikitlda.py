import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import cross_validation, linear_model
import numpy as np
from sklearn.preprocessing import Imputer
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

from matplotlib.colors import ListedColormap

def plot_decision_regions(X, y, classifier, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = 0 - 1, len(X) + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=0, y=X[y == cl, 0],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)
					
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

X_train, X_test ,y_train,y_test = cross_validation.train_test_split(X,y,test_size=0.2)

# define the object
stdsc = StandardScaler()


X_train_std= stdsc.fit_transform(X_train)

# once it learns it can apply on other inputs
X_test_std= stdsc.transform(X_test)

lda=LinearDiscriminantAnalysis(n_components=2)
X_train_lda=lda.fit_transform(X_train_std,y_train)

lr=LogisticRegression()
lr=lr.fit(X_train_lda,y_train)




print X_train_lda
plot_decision_regions(X_train_lda, y_train, classifier=lr)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower left')
plt.tight_layout()
# plt.savefig('./images/lda3.png', dpi=300)
plt.show()