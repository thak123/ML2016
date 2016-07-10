import pandas as pd
import numpy as np
from sklearn import preprocessing ,cross_validation,neighbors
from Classifier import Classifier
import os,sys,pickle
from sklearn.linear_model import LogisticRegression



def run():
	clf= Classifier('breast-cancer-wisconsin.data.txt')
	clf.clf_fit_transform()
	
	clf.default_accuracy_lr()
	
	clf.weight_coefficent_lr()
	clf.SBS_lf()
	# it takes long time to run the Random forest Code ...uncomment to check the result
	# clf.feature_selection_rf()
	clf.PCA()
	clf.pipe_kf_validation()
	clf.clf_learning_curve()
	clf.clf_validation_curve()
	clf.clf_roc_curve()
	
	clf.train()
	clf.l1l2()
	
	
	dest = os.path.join('classifier', 'pkl_objects')
	if not os.path.exists(dest):
		os.makedirs(dest)
	pickle.dump(clf,	open(os.path.join(dest, 'classifier.pkl'), 'wb'),	protocol=2)

	


if __name__ == "__main__":
	run()