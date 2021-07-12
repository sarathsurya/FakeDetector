from flask import Flask, render_template, request
import pickle
import pandas as pd

# create Flask application
app = Flask(__name__)

# read object TfidfVectorizer and model from disk
vec = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
@app.route('/first') 
def first():
	return render_template('first.html')
@app.route('/login') 
def login():
	return render_template('login.html')    
@app.route('/chart') 
def chart():
	return render_template('chart.html')    
@app.route('/abstract') 
def abstract():
	return render_template('first.html') 
@app.route('/future') 
def future():
	return render_template('future.html')  
@app.route('/upload') 
def upload():
	return render_template('upload.html') 
@app.route('/preview',methods=["POST"])
def preview():
    if request.method == 'POST':
        dataset = request.files['datasetfile']
        df = pd.read_csv(dataset,encoding = 'unicode_escape')
        df.set_index('Id', inplace=True)
        return render_template("preview.html",df_view = df)    

 
@app.route('/home')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    error = None
    if request.method == 'POST':
        # message
        msg = request.form['message']
        msg = pd.DataFrame(index=[0], data=msg, columns=['data'])

        # transform data
        text = vec.transform(msg['data'].astype('U'))
        # model
        result = model.predict(text)

        if result == 0:
            result = "real"
        else:
            result = 'fake'

        return render_template('index.html', prediction_value=result)
    else:
        error = "Invalid message"
        return render_template('index.html', error=error)


if __name__ == "__main__":
    app.run(debug=True)
