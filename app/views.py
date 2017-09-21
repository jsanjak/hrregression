from flask import Flask, render_template, request,session
from app import app
import pandas as pd
import numpy as np
from math import pi
#from textwrap import wrap
from bokeh.plotting import figure
from bokeh.embed import components
from bokeh.models import Span, FactorRange
from bokeh.charts import Bar
from scipy import stats
from sklearn.externals import joblib
from app.utils import *
from bokeh.io import output_file, show
from bokeh.layouts import widgetbox
from bokeh.models.widgets import Select
#from utils import make_histogram, plot_lasso, my_z_score, rank_direction


HF_env = joblib.load('HF_lasso_results.pkl') 
#####
#app = Flask(__name__)
app.secret_key = "super secret key"


############Intro to app section
@app.route('/')
@app.route('/index')
@app.route('/returns_input')
def returns_input():
	#HF_env['hospital_names'].values()#
	return(render_template("returns_input.html",hospitals = sorted(HF_env['hospital_names'].values())))


@app.route('/returns_predictors')
def returns_predictors():
	#get the relevant information
	current_feature_name = request.args.get("feature_name")
	if current_feature_name == None:
		current_feature_name = 'Staff responsiveness'#request.args.get("feature_name")

	current_provider_name = session.get('provider',None)
	if current_provider_name == None:
		current_provider_name = request.args.get("provider")
		session['provider'] = current_provider_name

	provider = HF_env['provider_id'][current_provider_name]#	
	rankmat = HF_env['model_mat'].rank(axis=0)/len(HF_env['model_mat'])

	#get the lasso model results
	lasso_results = HF_env['model_results'].loc[
	HF_env['model_results']['coef'].values!=0].loc[
	np.logical_not(
	HF_env['model_results']['measure_id'].isin(
	['H_RECMND_LINEAR_SCORE','MORT_30_COPD','MORT_30_HF']))]

	lasso_results['ranks'] = [rankmat.loc[int(provider),item] for 
	item in lasso_results['measure_id']]

	lasso_results['rank_coef'] = [rank_direction(i,j) for 
	i,j in zip(lasso_results['coef'],lasso_results['ranks'])]

	lasso_results['measure_names'] = [HF_env['measure_name_dict'][ID] for 
	ID in lasso_results['measure_id']]
	measure_id_dict = dict(zip(lasso_results['measure_names'],
		lasso_results['measure_id']))

	#make a select barm menu
	menu = [i for i in lasso_results['measure_names']]
	#select = Select(title="Indicator:",
	# value='Median ED to admission time',
	# options=menu)
	
	#drop_script, drop_div = components(widgetbox(select))

	#bar plots
	bar = plot_lasso(lasso_results)

	#histograms
	hist = make_histogram(HF_env['model_mat'].loc[:,measure_id_dict[current_feature_name]],int(provider),
		HF_env['hospital_names'][int(provider)].title(),
		str(current_feature_name)+ ' Score')
	hist_script, hist_div = components(hist)


	#script and div to send
	improvement_script, improvement_div = components(bar)
		
	#['Staff responsiveness','Discharge information'],
	return render_template('data.html',
		provider_name=HF_env['hospital_names'][int(provider)].title(),
		readmin_rate='heart failure return days', 
		feature_names= menu, 
		current_feature_name=current_feature_name,
		improvement_script=improvement_script, improvement_div=improvement_div,
		hist_script=hist_script, hist_div=hist_div)
	



