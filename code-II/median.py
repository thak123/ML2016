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
imr = Imputer(missing_values='NaN', strategy='most_frequent', axis=0,copy=False)

imr.fit(df)
X_imputed_df = pd.DataFrame(imr.transform(df.values), columns = df.columns)


X_imputed_df.drop(['id'],1,inplace=True)


X= np.array(X_imputed_df.drop(['class'],1))
y=np.array(X_imputed_df['class'])

le= LabelEncoder()
y=le.fit_transform(y)


#cross validation
X_train, X_test ,y_train,y_test = cross_validation.train_test_split(X,y,test_size=0.2)
X_train, X_test ,y_train,y_test=cross_validation.train_test_split(X,y,test_size=0.2)
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