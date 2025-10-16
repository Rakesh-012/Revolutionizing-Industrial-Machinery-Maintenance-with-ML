import flask
from flask import Flask, request, render_template, jsonify
import pickle
import pandas as pd
import numpy as np
import os
import traceback
import shap
import json

MODEL_FILE = 'model.pkl'
SHAP_BACKGROUND_DATA_FILE = 'shap_background_data.csv' 
TEMPLATES_DIR = 'templates'
STATIC_DIR = 'static' 

app = Flask(__name__, template_folder=TEMPLATES_DIR, static_folder=STATIC_DIR)

pipeline = None
preprocessor = None 
model = None      
preprocessed_feature_names = None 

if os.path.exists(MODEL_FILE):
    print(f"Loading model from {MODEL_FILE}...")
    try:
        with open(MODEL_FILE, 'rb') as f:
            pipeline = pickle.load(f)

        if hasattr(pipeline, 'steps') and len(pipeline.steps) > 1:
            preprocessor = pipeline.steps[0][1] 
            model = pipeline.steps[-1][1]     
            print("Pipeline, preprocessor, and model loaded successfully.")

            try:
                preprocessed_feature_names = preprocessor.get_feature_names_out()
                print("Preprocessed feature names obtained.")
            except Exception as e:
                print(f"Could not get preprocessed feature names: {e}")
                print("Using original feature names for SHAP plot (may be misleading after OHE).")
                preprocessed_feature_names = None 

        else:
             print("Warning: Loaded pipeline does not have expected structure (multiple steps). Cannot extract preprocessor/model.")
             model = pipeline 
             print("Assuming model expects raw numerical data or handles preprocessing internally.")
             preprocessed_feature_names = None


    except Exception as e:
        print(f"Error loading model file '{MODEL_FILE}': {e}")
        traceback.print_exc()
        pipeline = None 
        preprocessor = None
        model = None
else:
    print(f"Warning: Model file '{MODEL_FILE}' not found.")
    print("Prediction feature will be unavailable.")


background_data_raw = None
background_data_preprocessed = None 

if os.path.exists(SHAP_BACKGROUND_DATA_FILE):
    print(f"Loading SHAP background data from {SHAP_BACKGROUND_DATA_FILE}...")
    try:
        background_data_raw = pd.read_csv(SHAP_BACKGROUND_DATA_FILE)
        print("Raw SHAP background data loaded successfully.")

        if preprocessor is not None:
             print("Preprocessing background data...")
             background_data_preprocessed = preprocessor.transform(background_data_raw)
             print("Background data preprocessed.")
            
        elif model is not None:
             print("Warning: Preprocessor not found in pipeline. Assuming model can handle raw background data format.")
             background_data_preprocessed = background_data_raw.select_dtypes(include=np.number).to_numpy() 


    except Exception as e:
        print(f"Error processing SHAP background data file '{SHAP_BACKGROUND_DATA_FILE}': {e}")
        traceback.print_exc()
        background_data_preprocessed = None
else:
    print(f"Warning: SHAP background data file '{SHAP_BACKGROUND_DATA_FILE}' not found.")
    print("SHAP explanations will be unavailable.")

explainer = None

if model is not None and background_data_preprocessed is not None:
    try:
        if hasattr(model, 'predict_proba'):
             def predict_proba_fn(X_preprocessed):
                 return model.predict_proba(X_preprocessed)
             print("Using model.predict_proba for SHAP explainer.")
             shap_predict_fn = predict_proba_fn
        elif hasattr(model, 'decision_function'):
             def decision_function_fn(X_preprocessed):
                 return model.decision_function(X_preprocessed)
             print("Using model.decision_function for SHAP explainer.")
             shap_predict_fn = decision_function_fn
        elif hasattr(model, 'predict'):
             def predict_fn(X_preprocessed):
                  return model.predict(X_preprocessed)
             print("Warning: predict_proba/decision_function not found. Using model.predict for SHAP explainer.")
             shap_predict_fn = predict_fn
        else:
             raise AttributeError("Model does not have predict_proba, decision_function, or predict method.")


        print("Initializing SHAP Explainer...")
        if preprocessed_feature_names is not None:
             explainer = shap.Explainer(shap_predict_fn, background_data_preprocessed, feature_names=list(preprocessed_feature_names)) 
        else:
             explainer = shap.Explainer(shap_predict_fn, background_data_preprocessed)

        print("SHAP Explainer initialized successfully.")

    except Exception as e:
        print(f"Error initializing SHAP Explainer: {e}")
        traceback.print_exc()
        explainer = None 

failure_mapping_output = {
    0: "No Failure ✅",
    1: "Heat Dissipation Failure 🔥",
    2: "Power Failure ⚡",
    3: "Overstrain Failure 💪",
    4: "Tool Wear Failure 🔧",
    5: "Random Failures ❓" 
}

feature_columns = [
    'Type', 'Air temperature [K]', 'Process temperature [K]',
    'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]'
]


@app.route('/')
def home():
    """Renders the home page with the input form."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles the prediction request and generates SHAP explanation."""
    if pipeline is None:
        return render_template('index.html',
                               prediction_text='Error: Prediction model is not loaded. Cannot make predictions.',
                               error=True)

    if request.method == 'POST':
        shap_values_json = None
        shap_feature_names_json = None 
        shap_error = None
        prediction_text_display = None
        is_error_prediction = False

        try:
            machine_type = str(request.form['Type'])
            air_temp = float(request.form['Air temperature [K]'])
            process_temp = float(request.form['Process temperature [K]'])
            rotational_speed = float(request.form['Rotational speed [rpm]'])
            torque = float(request.form['Torque [Nm]'])
            tool_wear = float(request.form['Tool wear [min]'])

            input_data_raw = [[
                machine_type, air_temp, process_temp,
                rotational_speed, torque, tool_wear
            ]]
            input_df_raw = pd.DataFrame(input_data_raw, columns=feature_columns)

            print(f"Input DataFrame for prediction:\n{input_df_raw}")
            prediction_code = pipeline.predict(input_df_raw)[0] 

            prediction_label = failure_mapping_output.get(prediction_code, f"Unknown Failure Code ({prediction_code}) ❓")
            prediction_text_display = f'Predicted Failure Type: {prediction_label}'
            is_error_prediction = (prediction_code != 0)
            print(f"Prediction code: {prediction_code}, Prediction text: {prediction_label}")


            if explainer is not None and preprocessor is not None: 
                try:
                    print("Preprocessing input data for SHAP...")
                    input_data_preprocessed = preprocessor.transform(input_df_raw)
                    print("Input data preprocessed for SHAP.")

                    print("Calculating SHAP values...")
                    shap_values_obj = explainer(input_data_preprocessed)
                    print("SHAP values calculated.")

                    if hasattr(shap_values_obj, 'values') and isinstance(shap_values_obj.values, np.ndarray):
                         if len(shap_values_obj.values.shape) == 3: 
                             if prediction_code < shap_values_obj.values.shape[2]:
                                 sv_for_prediction = shap_values_obj.values[0, :, prediction_code]
                                 print(f"Extracted SHAP values for predicted class index {prediction_code}.")
                             else:
                                  shap_error = f"Predicted class index {prediction_code} is out of bounds for SHAP values (shape {shap_values_obj.values.shape})."
                                  print(shap_error)
                                  sv_for_prediction = None 
                         elif len(shap_values_obj.values.shape) == 2: 
                             print("Warning: SHAP values shape is 2D. Interpreting as importance for the model's primary output.")
                             sv_for_prediction = shap_values_obj.values[0, :]
                             if prediction_code > 1:
                                 print(f"Note: SHAP 2D output used for predicted class {prediction_code}. Explanation might be less accurate than for binary cases.")

                         else:
                              raise ValueError(f"Unexpected SHAP values shape: {shap_values_obj.values.shape}")

                         if sv_for_prediction is not None:
                              if preprocessed_feature_names is not None:
                                  feature_names_for_chart = list(preprocessed_feature_names)
                                  print("Using preprocessed feature names for chart.")
                              else:
                                  feature_names_for_chart = feature_columns
                                  print("Using original feature names for chart (preprocessed names not available).")


                              shap_values_json = json.dumps(sv_for_prediction.tolist())
                              shap_feature_names_json = json.dumps(feature_names_for_chart)
                              print("SHAP data prepared for template.")
                         else:
                              shap_error = shap_error if shap_error else "Could not extract valid SHAP values."


                    else:
                         print("Warning: Could not extract numeric SHAP values from explainer output (no .values or not numpy array).")
                         shap_error = "Could not extract SHAP values from explainer output."


                except Exception as e:
                    print(f"Error during SHAP calculation: {e}")
                    traceback.print_exc()
                    shap_error = f"Could not generate explanation: {e}"
            elif explainer is None:
                shap_error = "SHAP explainer not available."
                print("Skipping SHAP calculation: Explainer not initialized.")
            elif preprocessor is None:
                 shap_error = "Preprocessor not available. Cannot preprocess input for SHAP."
                 print("Skipping SHAP calculation: Preprocessor not found in pipeline.")


            return render_template('index.html',
                                   prediction_text=prediction_text_display,
                                   error=is_error_prediction,
                                   shap_values=shap_values_json,
                                   shap_features=shap_feature_names_json,
                                   shap_error=shap_error,
                                   form_data=request.form)


        except FileNotFoundError: 
            print(f"Error: Model file '{MODEL_FILE}' not found during prediction.")
            return render_template('index.html', prediction_text='Error: Model file not found. Cannot predict.', error=True)
        except KeyError as e:
            print(f"Error: Missing form field - {e}")
            return render_template('index.html', prediction_text=f'Error: Missing input field: {e}. Please fill all fields.', error=True)
        except ValueError as e:
            print(f"Error: Invalid input value - {e}")
            return render_template('index.html', prediction_text='Error: Please enter valid numbers for temperature, speed, torque, and wear.', error=True)
        except Exception as e:
            print(f"An unexpected error occurred during prediction: {e}")
            traceback.print_exc()
            return render_template('index.html', prediction_text=f'An error occurred during prediction: {e}. Check server logs.', error=True)

    return flask.redirect(flask.url_for('home'))


@app.route('/chat', methods=['POST'])
def chat_handler():
    """Handles incoming chat messages and returns a response based on keyword mapping."""
    if not request.is_json:
        print("Chat Error: Request was not JSON")
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    user_message = data.get('message')

    if not user_message:
        print("Chat Error: 'message' key missing from JSON")
        return jsonify({"error": "Missing 'message' key in JSON data"}), 400

    print(f"Received chat message: {user_message}")

    user_message_lower = user_message.lower()

    responses = {
        ("hello", "hi", "hey"): "Hello there! How can I assist with the machine failure prediction today?",
        ("help", "assist", "support"): "Sure! You can enter the machine parameters in the form to get a failure prediction. You can also ask me about specific parameters or failure types.",
        ("air temperature", "air temp", "ambient temp"): "Air temperature (in Kelvin) is the ambient temperature around the machine. It can affect heat dissipation.",
        ("process temperature", "process temp", "operational temp"): "Process temperature (in Kelvin) is the temperature generated during the machine's operation. High process temps can indicate issues.",
        ("rotational speed", "speed", "rpm"): "Rotational speed (in RPM) indicates how fast the machine's spindle is rotating. Unusual speeds can be linked to failures.",
        ("torque", "rotational force"): "Torque (in Newton-meters) is the rotational force applied by the machine. High torque can lead to overstrain.",
        ("tool wear", "wear"): "Tool wear (in minutes) measures the operational time the tool has been used. Excessive wear increases the chance of failure.",
        ("machine type", "quality"): "The machine type refers to its quality variant (Low, Medium, High - L, M, H). This can influence its general reliability.",
        ("heat dissipation", "hdf"): "Heat Dissipation Failure (HDF) occurs when the machine overheats, often due to a large difference between air and process temperature combined with low rotational speed.",
        ("power failure", "pwf", "motor struggle"): "Power Failure (PWF) can happen when the torque is high and the rotational speed is low, indicating the motor is struggling.",
        ("overstrain failure", "osf", "overstress"): "Overstrain Failure (OSF) typically results from a combination of high tool wear and high torque, putting too much stress on the components.",
        ("tool wear failure", "twf"): "Tool Wear Failure (TWF) is directly related to the tool reaching its operational limit, indicated by high 'Tool wear [min]' values.",
        ("no failure", "normal operation"): "That's great! 'No Failure' means the current parameters suggest the machine is operating normally.",
        ("explain", "shap", "importance", "chart", "graph"): ("The 'Prediction Explanation' chart (if shown) uses SHAP values to show how much each input parameter "
                                                                "contributed to the prediction. Red bars mean the feature pushed the prediction towards failure, "
                                                                "while blue bars pushed it towards 'No Failure'. Longer bars have a bigger impact."
                                                                "The feature names might look slightly different if they were processed (e.g., 'Type_L' for Machine Type 'L')."),
        ("thank you", "thanks", "thank you very much"): "You're welcome! Let me know if you have more questions.",
        ("bye", "goodbye", "see you"): "Goodbye! Feel free to ask if you need help again."
    }

    for trigger_phrases, reply in responses.items():
        sorted_phrases = sorted(trigger_phrases, key=len, reverse=True)
        for phrase in sorted_phrases:
            if phrase in user_message_lower:
                print(f"Matched phrase: '{phrase}'")
                return jsonify({"reply": reply})

    bot_reply = "Sorry, I didn't quite understand that. Can you rephrase? You can ask about specific parameters or failure types."
    print(f"No specific match found. Sending default reply.")
    return jsonify({"reply": bot_reply})
    print(f"Sending chat reply: {bot_reply}")
    return jsonify({"reply": bot_reply})


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True) 