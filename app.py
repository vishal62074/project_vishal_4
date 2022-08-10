import numpy as np
from flask import Flask, request, jsonify, render_template

import pickle


app = Flask(__name__)
model1 = pickle.load(open('Titanic_modelKNN.pkl','rb')) 


@app.route('/')
def home():
  
    return render_template("index.html")
  
@app.route('/predict',methods=['GET'])

def predict():
    
    
    '''
    For rendering results on HTML GUI
    '''
    Pclass = float(request.args.get('Pclass'))
    Age = float(request.args.get('Age'))
    SibSp = float(request.args.get('SibSp'))
    Parch = float(request.args.get('Parch'))
    Fare = float(request.args.get('Fare'))
    Sex = float(request.args.get('Sex'))

    
    prediction = model1.predict([[Pclass, Sex, Age, SibSp, Parch, Fare]])
    if prediction == [1]:
      prediction_text = 'Survived'
    else:
      prediction_text = 'Not Survived'
        
    return render_template('index.html', prediction_text='KNN model has predict about the survival of person for given features  : {}'.format(prediction_text))


if __name__=="__main__":
  app.run(debug=True)
