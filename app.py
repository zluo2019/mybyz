from flask import Flask, render_template,request
import pandas as pd
from transformer import Encode_Transformer
import pickle

app = Flask(__name__)

model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/", methods = ['POST'])
def predict():

    X = {'person_income': [float(request.form["person_income"])],
        'person_home_ownership': [request.form["person_home_ownership"]],
        'person_emp_length': [float(request.form["person_emp_length"])],
        'loan_intent': [request.form["loan_intent"]],
        'loan_grade': [request.form["loan_grade"]],
        'loan_percent_income': [float(request.form["loan_percent_income"])],
        'cb_person_default_on_file': [request.form["cb_person_default_on_file"]]}

    X = pd.DataFrame(data=X)
    transformer = Encode_Transformer()
    X = transformer.transform(X)    
    
    print(X)
    prediction = model.predict(X)
    
    if prediction == 1:
        return render_template("index.html", prediction_text_1 = "This customer will default")
    else:
        return render_template("index.html", prediction_text_0 = "This customer will NOT default")

if __name__ == '__main__':
    app.run(debug=True)