from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
try:
    model = joblib.load('/home/zenkilogram/mysite/random_forest_full.pkl')
except FileNotFoundError:
    model = None
    print("Error: Model file 'random_forest_full.pkl' not found.")

@app.route('/')
def home():
    return render_template('homepage.html')  

@app.route('/templates/index.html')
def newindex():
    return render_template("index.html")

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded. Check the server logs for more details.'}), 500

    try:
        # Get JSON data from the request
        data = request.get_json()

        # Validate input data
        required_features = [
            'age', 'sex', 'cp', 'trtbps', 'chol', 'fbs', 'restecg',
            'thalach', 'exng', 'oldpeak', 'slp', 'caa', 'thall'
        ]
        if not all(feature in data for feature in required_features):
            return jsonify({'error': 'Missing one or more required features'}), 400

        # Prepare data for prediction
        try:
            features = [float(data[feature]) for feature in required_features]
        except ValueError:
            return jsonify({'error': 'All features must be numeric.'}), 400

        features_array = np.array(features).reshape(1, -1)  # Reshape to (1, 13)

        # Make prediction
        prediction = model.predict(features_array)

        # Return prediction result
        result = {
            'prediction': int(prediction[0])  # Convert numpy.int64 to int for JSON serialization
        }
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the app (this is only for local testing)
if __name__ == '__main__':
    app.run(debug=True)
