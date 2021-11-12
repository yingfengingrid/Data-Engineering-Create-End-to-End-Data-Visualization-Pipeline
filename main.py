#!/usr/bin/env python
from flask import Flask, render_template, flash, request, jsonify, Markup
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io, base64, os
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from PIL import Image
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
 

# default traveler constants
DEFAULT_PLOT = 'Pairplot'
DEFAULT_SEPWIDTH = 3.3
DEFAULT_SEPLEN = 3.3
DEFAULT_PETWIDTH = 3.3
DEFAULT_PETLEN = 3.3


# # initializing constant vars
# average_survival_rate = 0
# # logistic regression modeling
# lr_model = LogisticRegression()

app = Flask(__name__)


@app.before_first_request
def startup():
    global model,iris

    iris = pd.read_csv('iris.csv', usecols=[1, 2, 3, 4, 5])
    model=KNeighborsClassifier(n_neighbors=3)
    X = iris[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
    y = iris['Species']
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)
    model.fit(X, y)
    
    


@app.route("/", methods=['POST', 'GET'])
def submit_new_profile():
    model_results = ''
    if request.method == 'POST':
        selected_plot = request.form['selected_plot']
        PetalWidth = request.form['PetalWidth']
        PetalLength = request.form['PetalLength']
        SepalLength = request.form['SepalLength']
        SepalWidth = request.form['SepalWidth']
        antV = ['#1890FF', '#2FC25B', '#FACC14', '#223273', '#8543E0', '#13C2C2', '#3436c7', '#F04864'] 
        
        if selected_plot=='Pairplot':
            fig = sns.pairplot(data=iris, palette=antV[0:3], hue= 'Species')
      
            img = io.BytesIO()
            fig.savefig(img, format='png')
            img.seek(0)
            plot_url = base64.b64encode(img.getvalue()).decode()

            
            
            
            
        if selected_plot=='Andrews Curves':
            plt.figure(figsize = (10,8))
            pd.plotting.andrews_curves(iris, 'Species', colormap='cool')
            fig=plt.gcf()

            img = io.BytesIO()
            fig.savefig(img, format='png')
            plot_url = base64.b64encode(img.getvalue()).decode()
            
        if selected_plot=='Linear regression':
            fig=sns.lmplot(data=iris, x='SepalWidthCm', y='SepalLengthCm', palette=antV, hue='Species')
            
            img = io.BytesIO()
            fig.savefig(img, format='png')
            img.seek(0)
            plot_url = base64.b64encode(img.getvalue()).decode()
        
        test_X=[[float(SepalLength),float(SepalWidth),float(PetalLength),float(PetalWidth)]]
        pred=model.predict(test_X)
        if pred[0]==0:
            kind=r'static\images\setosa.jpg'
        elif pred[0]==1:
            kind=r'static\images\versicolour.jpg'
        else:
            kind=r'static\images\virginica.jpg'
        
        plt.figure(figsize=(6,6))
        img=Image.open(kind)
        plt.imshow(img)
        fig=plt.gcf()
        img = io.BytesIO()
        fig.savefig(img, format='png')
        plot_url2 = base64.b64encode(img.getvalue()).decode()
        
            
        print(pred)
        
        return render_template('index.html',
            model_results = Markup('<img src="data:image/png;base64,{}">'.format(plot_url2)),
            model_plot = Markup('<img src="data:image/png;base64,{}">'.format(plot_url)),
            selected_plot=DEFAULT_PLOT,
            PetalWidth=DEFAULT_PETWIDTH,
            PetalLength=DEFAULT_PETLEN,
            SepalLength=DEFAULT_SEPLEN,
            SepalWidth=DEFAULT_SEPWIDTH)
    else:
        # set default passenger settings
        return render_template('index.html',
            model_results = '',
            model_plot = '',
            selected_plot=DEFAULT_PLOT,
            PetalWidth=DEFAULT_PETWIDTH,
            PetalLength=DEFAULT_PETLEN,
            SepalLength=DEFAULT_SEPLEN,
            SepalWidth=DEFAULT_SEPWIDTH)




    #     except FileNotFoundError:
    #        return render_template('index.html',
    #         model_results = '',
    #         model_plot = '',
    #         selected_plot=DEFAULT_PLOT,
    #         PetalWidth=DEFAULT_PETWIDTH,
    #         PetalLength=DEFAULT_PETLEN,
    #         SepalLength=DEFAULT_SEPLEN,
    #         SepalWidth=DEFAULT_SEPWIDTH)
           
    # else:
    #     # set default passenger settings
    #     return render_template('index.html',
    #         model_results = '',
    #         model_plot = '',
    #         selected_plot=DEFAULT_PLOT,
    #         PetalWidth=DEFAULT_PETWIDTH,
    #         PetalLength=DEFAULT_PETLEN,
    #         SepalLength=DEFAULT_SEPLEN,
    #         SepalWidth=DEFAULT_SEPWIDTH)

if __name__=='__main__':
	app.run(debug=False)