import numpy as np
import pickle
import pandas as pd
from flask import Flask, request, render_template

app = Flask(__name__)

import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, render_template

app = Flask(__name__)

# Load model, scaler, and feature names
model = pickle.load(open("model.pkl", "rb"))
scale = pickle.load(open("scale.pkl", "rb"))
feature_names = pickle.load(open("features.pkl", "rb"))  # <- Load saved feature names

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=["POST"])
def predict():
    try:
        input_features = [float(x) for x in request.form.values()]
        df = pd.DataFrame([np.array(input_features)], columns=feature_names)
        df_scaled = scale.transform(df)
        prediction = model.predict(df_scaled)
        result = f"Estimated Traffic Volume is: {int(prediction[0])}"
        return render_template("index.html", prediction_text=result)
    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
