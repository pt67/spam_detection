from flask import Flask, render_template, request
import pickle

# Load only the full pipeline (vectorizer + model)
with open('spam_pipeline.pkl', 'rb') as f:
    pipeline = pickle.load(f)

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']
    prediction = pipeline.predict([message])[0]
    result = "SPAM" if prediction == 1 else "HAM (Not Spam)"
    return render_template('index.html', prediction=result, message=message)

if __name__ == '__main__':
    app.run(debug=True)
