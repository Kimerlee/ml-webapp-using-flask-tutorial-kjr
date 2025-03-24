from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import os


file_path = os.path.join(os.path.dirname(__file__), 'global_food_wastage_dataset.pkl')

with open(file_path, 'rb') as file:
    model = pickle.load(file)


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        population = float(request.form['population'])
        economic_loss = float(request.form['economic_loss'])
        avg_waste_per_capita = float(request.form['avg_waste_per_capita'])
        household_waste = float(request.form['household_waste'])

        # Create input array for model
        input_data = np.array([[population, economic_loss, avg_waste_per_capita, household_waste]])
        
        # Make prediction
        prediction = model.predict(input_data)

        return render_template('index.html', prediction_text=f'Predicted Waste: {prediction[0]:.2f} tons')

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)
