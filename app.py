from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
from flask_cors import CORS
# After creating app


app = Flask(__name__)

CORS(app)
# Load Scaler and Model
scaler = joblib.load('ScalerNew.pkl')
model = joblib.load('ridge_model.pkl')
@app.route('/')
def home():
    return render_template('newIndex.html')
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json['input']
        input_array = np.array(data).reshape(1, -1)

        # Step 1: Scale the input
        scaled_input = scaler.transform(input_array)

        # Step 2: Predict using the ML model
        prediction = model.predict(scaled_input)
        array=prediction[0]
        print("hek")
        print(prediction)
        return jsonify({
    'pred1': float(array[0]),
    'pred2': float(array[1]),
    'pred3': float(array[2])
})


    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
