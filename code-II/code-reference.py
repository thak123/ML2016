import pickle


clf= LinearRegression(n_jobs=-1)
clf.fit(X_train,y_train)

#save to pickle
with open('linearregression.pickle','wb') as f:
	pickle.dump(clf,f)
	
#load from pickle
pickle_in = open('linearregression.pickle','rb')
clf=pickle.load(pickle_in)