from flask import Flask, render_template, request,session
from app import app
import pandas as pd
import numpy as np
from math import pi
from textwrap import wrap
from bokeh.plotting import figure
from bokeh.embed import components
from bokeh.models import Span, FactorRange
from bokeh.charts import Bar
from scipy import stats
from sklearn.externals import joblib



HF_env = joblib.load('HF_lasso_results.pkl') 
#####
#app = Flask(__name__)
#app.secret_key = "super secret key"


############Intro to app section
@app.route('/')
@app.route('/index')
@app.route('/returns_input')
def returns_input():
	#HF_env['hospital_names'].values()#
	return(render_template("returns_input.html",hospitals = sorted(HF_env['hospital_names'].values())))


def plot_lasso(lasso_results):
	data = lasso_results.sort_values(by='rank_coef',ascending=True)
	yrange = data['measure_names'].tolist()

	p = figure(width=800, height=600, x_range =[0,1.1*np.max(data['rank_coef'].values)], y_range=yrange)

	p.rect(x=data['rank_coef']/2, y=yrange, 
		width=abs(data['rank_coef']), height=0.4,color=(76,114,176),
    width_units="data", height_units="data")


	#p = Bar(data, 
	#	'measure_names', values='rank_coef', legend=False,
	#	title="Most Actionable Factors")
	#p.x_range = FactorRange(factors=data['measure_names'])
	p.yaxis.major_label_orientation = pi/12
	p.yaxis.axis_label = None
	p.xaxis.axis_label = 'Actionability Index'
	p.xaxis.axis_label_text_font_size = "20pt" 
	p.yaxis.major_label_text_font_size = "10pt"
	p.yaxis.axis_label_text_font_size = "20pt" 
	return(p)

def my_z_score(x):
    z=((x.values - x.mean())/x.std())
    return(z)


def rank_direction(coef,rank):
	if coef <0:
		rank_coef = (-1*coef)/(1 - rank)
	else:
		rank_coef = coef/rank
	return(rank_coef)

@app.route('/returns_predictors')
def returns_predictors():
	#readmin_rate=request.args.get('readmin_rate')
	provider = HF_env['provider_id'][request.args.get('provider')]#session.get('provider',None)

	#SQLdata = MakeSQLData(readmin_rate).SQLdata
	#SQLdata.rank(axis=0)/len(SQLdata)		
	rankmat = HF_env['model_mat'].rank(axis=0)/len(HF_env['model_mat'])

	lasso_results = HF_env['model_results'].loc[
	HF_env['model_results']['coef'].values!=0].loc[
	np.logical_not(
	HF_env['model_results']['measure_id'].isin(
	['H_RECMND_LINEAR_SCORE','MORT_30_COPD','MORT_30_HF']))]

	#train_lasso(SQLdata)
	lasso_results['ranks'] = [rankmat.loc[int(provider),item] for 
	item in lasso_results['measure_id']]

	lasso_results['rank_coef'] = [rank_direction(i,j) for 
	i,j in zip(lasso_results['coef'],lasso_results['ranks'])]

	lasso_results['measure_names'] = [HF_env['measure_name_dict'][ID] for 
	ID in lasso_results['measure_id']]

	bar = plot_lasso(lasso_results)

	improvement_script, improvement_div = components(bar)

	return render_template('data.html',provider=HF_env['hospital_names'][int(provider)].title(),
		readmin_rate='heart failure return days',
		improvement_script=improvement_script, improvement_div=improvement_div)

#if __name__ == '__main__':
#  app.run(host='0.0.0.0',debug=True, port=8000)

#hist = make_histogram(HF_env['model_mat'].iloc[:,0],provider,'Scaled heart failure return days')
#hist_script, hist_div = components(hist)
	



