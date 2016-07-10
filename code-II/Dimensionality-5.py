import pandas as pd
import matplotlib.pyplot as plt
import numpy  as np
from sklearn import cross_validation
from sklearn.preprocessing import Imputer,StandardScaler


df= pd.read_csv('breast-cancer-wisconsin.data.txt',na_values=["?"])

imr = Imputer(missing_values='NaN', strategy='mean', axis=0,copy=False)
imr.fit(df)
X_imputed_df = pd.DataFrame(imr.transform(df.values), columns = df.columns)


X_imputed_df.drop(['id'],1,inplace=True)


X= np.array(X_imputed_df.drop(['class'],1))
y=np.array(X_imputed_df['class'])

X_train, X_test ,y_train,y_test = cross_validation.train_test_split(X,y,test_size=0.2,random_state=0)


# define the object
stdsc = StandardScaler()


X_train_std= stdsc.fit_transform(X_train)

# once it learns it can apply on other inputs
X_test_std= stdsc.transform(X_test)


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