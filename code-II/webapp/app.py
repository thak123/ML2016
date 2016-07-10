from flask import Flask, render_template, request
from wtforms import Form, validators
from wtforms.fields import IntegerField
# from wtforms.fields.html5 import DecimalRangeField,IntegerRangeField
import os,sys
import numpy as np
import pickle,inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

from Classifier import  Classifier
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Prepare the Classifier 
cur_dir = os.path.dirname(__file__)

# Unpickle the classifier
clf = pickle.load(open(os.path.join(parentdir,'classifier','pkl_objects','classifier.pkl'), 'rb'))




# In app function for calling Predict ????? Is it required
def predict(x):
	# x=np.array(x)
	return clf.predict_unseen(x)

class PredictForm(Form):	
	clump_thickness = IntegerField(label='Clump Thickness:',   validators=[validators.NumberRange(1, 10)])
	unif_cell_size = IntegerField(label='Unif Cell Size:',   validators=[validators.NumberRange(1, 10)])
	unif_cell_shape = IntegerField(label='Unif Cell Shape:',   validators=[validators.NumberRange(1, 10)])
	marg_adhesion = IntegerField(label='Marg Adhesion:',   validators=[validators.NumberRange(1, 10)])
	single_epith_cell_size = IntegerField(label='Single Epith Cell Size:',   validators=[validators.NumberRange(1, 10)])
	bare_nuclei = IntegerField(label='Bare Nuclei:',   validators=[validators.NumberRange(1, 10)])
	bland_chrom = IntegerField(label='Bland Chrom:', validators=[validators.NumberRange(1, 10)])
	norm_nucleoli = IntegerField(label='Norm Nucleoli:', validators=[validators.NumberRange(1, 10)])
	mitoses = IntegerField(label='Mitoss:',validators=[validators.NumberRange(1, 10)])

@app.route('/')
def index():
    form = PredictForm(request.form)
    return render_template('predictform.html', form=form, z='')

@app.route('/results', methods=['POST'])
def results():    
	form = PredictForm(request.form)
	z = ''
	if request.method == 'POST' and form.validate():
		features= [request.form['clump_thickness'],request.form['unif_cell_size'],request.form['unif_cell_shape'] ,request.form['marg_adhesion'],request.form['single_epith_cell_size'],request.form['bare_nuclei'],request.form['bland_chrom'],request.form['norm_nucleoli'] ,request.form['mitoses']]
		
		z = []
		if(predict(features)[0][0]==0):
			z.append('Benign')
			z.append(np.max(predict(features)[1]))
		else :
			z.append('Malignant')
			z.append(np.max(predict(features)[1]))
	print form.errors
	return render_template('predictform.html', form=form, z=z)

if __name__ == '__main__':
    app.run(debug=True)