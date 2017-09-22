from bokeh.plotting import figure
from bokeh.embed import components
from bokeh.models import Span, FactorRange
from bokeh.charts import Bar

import pandas as pd
import numpy as np
from math import pi

def make_histogram(data,provider_id,provider_name,readmin_rate):
		provider_data=data.ix[provider_id]
		hist, edges = np.histogram(data, density=True, bins=50)
		p = figure()#title = "Histogram of heart return days with value for "+ provider_name + " marked in red" )
		p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
        fill_color="#036564", line_color="#033649")
		p.line([provider_data,provider_data],[0,.5],line_width=2,color="red")
		p.yaxis.axis_label = None
		p.xaxis.axis_label = readmin_rate
		p.xaxis.axis_label_text_font_size = "20pt" 
		p.yaxis.major_label_text_font_size = "10pt"
		p.yaxis.axis_label_text_font_size = "20pt" 
		return(p)

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

def scale_zero_one(data):
    return(0.5 + (data - np.min(data))/(np.max(data)-np.min(data)))

def rank_direction(coef,scaled):
	if coef >0:
		#delta_1 = scaled - 1  
		rank_coef = np.abs(coef)/(2 - scaled)
	else:
		rank_coef = np.abs(coef)/(scaled)
	return(rank_coef)


