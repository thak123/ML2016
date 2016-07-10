import matplotlib.pyplot as plt
import pandas as pd

import numpy as np
from sklearn.preprocessing import Imputer

from sklearn.preprocessing import StandardScaler
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression

from SBS import SBS 
df= pd.read_csv('breast-cancer-wisconsin.data.txt',na_values=["?"])

#use median,most_frequent,mean
imr = Imputer(missing_values='NaN', strategy='mean', axis=0,copy=False)

imr.fit(df)
X_imputed_df = pd.DataFrame(imr.transform(df.values), columns = df.columns)


X_imputed_df.drop(['id'],1,inplace=True)


X= np.array(X_imputed_df.drop(['class'],1))
y=np.array(X_imputed_df['class'])

X_train, X_test ,y_train,y_test = cross_validation.train_test_split(X,y,test_size=0.2)

# define the object
stdsc = StandardScaler()


X_train_std= stdsc.fit_transform(X_train)

# once it learns it can apply on other inputs
X_test_std= stdsc.transform(X_test)


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


lr.fit(X_train_std[:, k5], y_train)
print('Training accuracy:', lr.score(X_train_std[:, k5], y_train))
print('Test accuracy:', lr.score(X_test_std[:, k5], y_test))


for iterating_var in sbs.subsets_:
	k_iter=list(iterating_var)
	lr.fit(X_train_std[:, k_iter], y_train)
	# print('Index %d'%idx)
	print('Training accuracy:', lr.score(X_train_std[:, k_iter], y_train))
	print('Test accuracy:', lr.score(X_test_std[:, k_iter], y_test))
