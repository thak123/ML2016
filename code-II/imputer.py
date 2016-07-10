import pandas as pd

# checking for the null columns
df=pd.read_csv('breast-cancer-wisconsin.data.txt')
print df.isnull().sum()

df=pd.read_csv('wdbc.data.txt')
print df.isnull().sum()

df=pd.read_csv('wpbc.data.txt')
print df.isnull().sum()
#ends

