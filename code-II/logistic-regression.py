import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import cross_validation, linear_model
import numpy as np
from sklearn.preprocessing import Imputer
import matplotlib.pyplot as plt

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


lr= linear_model.LogisticRegression(penalty='l1',C=0.1)
lr.fit(X_train_std,y_train)


accuracy=lr.score(X_test_std,y_test)
print(accuracy)


fig = plt.figure()
ax = plt.subplot(111)
    
colors = ['blue', 'green', 'red', 'cyan', 
          'magenta', 'yellow', 'black', 
          'pink', 'lightgreen', 'lightblue', 
          'gray', 'indigo', 'orange']

weights, params = [], []
for c in np.arange(-4, 6):
    lr = linear_model.LogisticRegression(penalty='l1', C=10**c, random_state=0)
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

print(lr.coef_)