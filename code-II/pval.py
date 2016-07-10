import pandas as pd
import matplotlib.pyplot as plt
import numpy  as np
from sklearn import cross_validation
from sklearn.preprocessing import Imputer,StandardScaler


df= pd.read_csv('breast-cancer-wisconsin.data.txt',na_values=["?"])

imr = Imputer(missing_values='NaN', strategy='median', axis=0,copy=False)
imr.fit(df)
X_imputed_df = pd.DataFrame(imr.transform(df.values), columns = df.columns)


X_imputed_df.drop(['id'],1,inplace=True)


X= np.array(X_imputed_df.drop(['class'],1))
y=np.array(X_imputed_df['class'])

print np.corrcoef(X)
np.savetxt('np.out.txt', np.corrcoef(X),'%.4e',newline ='\n') 