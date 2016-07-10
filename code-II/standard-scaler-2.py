import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import cross_validation, neighbors
import numpy as np
from sklearn.preprocessing import Imputer

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


clf= neighbors.KNeighborsClassifier()
clf.fit(X_train_std,y_train)


accuracy=clf.score(X_test_std,y_test)
print(accuracy)