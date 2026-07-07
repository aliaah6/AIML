# Import the Flask class, request object to access incoming request data, jsonify for sending JSON responses, and render_template for rendering HTML files.
from flask import Flask, request, jsonify, render_template
# Import the pickle library to load the serialized machine learning model file.
import pickle
# Import numpy for numerical and array operations, specifically to format the inputs for model prediction.
import numpy as np

# Initialize the Flask application, setting the name of the module.
app = Flask(__name__)

# Open the serialized machine learning model file 'model.pkl' in binary read mode ('rb').
with open('model.pkl', 'rb') as file:
    # Load and deserialize the model object from the file using pickle.
    model = pickle.load(file)

# Define the root route of the web application that handles GET requests to display the main page.
@app.route('/')
# Define the function that runs when a user visits the root URL.
def home():
    # Render and return the main HTML template named 'index.html'.
    return render_template('index.html')

# Define the predict route that handles POST requests containing user input data for prediction.
@app.route('/predict', methods=['POST'])
# Define the function that processes inputs and computes the predicted insurance charges.
def predict():
    # Start a try block to handle any potential runtime errors during data parsing or prediction.
    try:
        # Extract the JSON data sent in the request body from the client's AJAX request.
        data = request.get_json()
        
        # Extract the numeric age value from the input data and convert it to an integer.
        age = int(data['age'])
        # Extract the BMI (Body Mass Index) from the input data and convert it to a float.
        bmi = float(data['bmi'])
        # Extract the number of children from the input data and convert it to an integer.
        children = int(data['children'])
        # Extract the sex input string and convert it to lowercase for comparison.
        sex = data['sex'].lower()
        # Extract the smoker status input string and convert it to lowercase for comparison.
        smoker = data['smoker'].lower()
        # Extract the region input string and convert it to lowercase for comparison.
        region = data['region'].lower()
        
        # Encode sex: sex_male is 1 if sex is 'male', and 0 if it is 'female'.
        sex_male = 1 if sex == 'male' else 0
        # Encode smoker: smoker_yes is 1 if smoker status is 'yes', and 0 if 'no'.
        smoker_yes = 1 if smoker == 'yes' else 0
        
        # Initialize the three region dummy variables (northwest, southeast, southwest) to 0.
        region_northwest = 0
        region_southeast = 0
        region_southwest = 0
        
        # If the region is northwest, set region_northwest dummy variable to 1.
        if region == 'northwest':
            region_northwest = 1
        # If the region is southeast, set region_southeast dummy variable to 1.
        elif region == 'southeast':
            region_southeast = 1
        # If the region is southwest, set region_southwest dummy variable to 1.
        elif region == 'southwest':
            region_southwest = 1
        # Note: If the region is northeast, all three dummy variables remain 0. This is the reference category.
        
        # Construct the 2D features array (list of lists) expected by scikit-learn's model.predict.
        # The features must be in the exact order the model was trained on:
        # ['age', 'bmi', 'children', 'sex_male', 'smoker_yes', 'region_northwest', 'region_southeast', 'region_southwest']
        features = [[
            age, 
            bmi, 
            children, 
            sex_male, 
            smoker_yes, 
            region_northwest, 
            region_southeast, 
            region_southwest
        ]]
        
        # Pass the constructed features array to the model's predict method to calculate the insurance charge.
        prediction = model.predict(features)
        
        # Extract the scalar prediction value from the returned numpy array (index 0).
        predicted_charge = float(prediction[0])
        
        # If the predicted charge is less than 0 (possible with linear regression on edge cases), set it to 0.0.
        predicted_charge = max(0.0, predicted_charge)
        
        # Return a JSON response with status 'success' and the predicted charge value.
        return jsonify({
            'status': 'success',
            'prediction': predicted_charge
        })
        
    # Catch any Exception (e.g. ValueError, KeyError) that occurs during data extraction or prediction.
    except Exception as e:
        # Return a JSON response with status 'error' and the description of the exception, along with an HTTP 400 Bad Request status.
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

# Check if this script is executed directly (not imported as a module).
if __name__ == '__main__':
    # Start the Flask development server on the local machine on port 5000 with debugging enabled.
    app.run(host='127.0.0.1', port=5000, debug=True)
