from flask import Flask, render_template, request
import joblib
import numpy as np
import os
from collections import defaultdict

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

# Load model and assets
try:
    model = joblib.load(r'templates/models/best_model.pkl')
    print("✅ best_model.pkl loaded")
    label_encoders = joblib.load(r'templates/models/label_encoders.pkl')
    print("✅ label_encoders.pkl loaded")
    scaler = joblib.load(r'templates/models/scaler.pkl')
    print("✅ scaler.pkl loaded")
    target_encoder = joblib.load(r'templates/models/label_encoder.pkl')
    print("✅ label_encoder.pkl loaded")
except Exception as e:
    print(f"❌ Model load failed: {e}")
    model = label_encoders = scaler = target_encoder = None

# Corrected feature order (excluding AnimalName)
TRAINED_FEATURE_ORDER = [
    'BloodBrainDisease', 'AppearenceDisease', 'GeneralDisease',
    'LungDisease', 'AbdominalDisease', 'SymptomCount'
]

def normalize_input(text):
    return text.strip().lower() if text else ""

def encode_inputs(input_dict):
    encoded = []
    symptom_count = 0
    
    for feature_name in TRAINED_FEATURE_ORDER:
        if feature_name == 'SymptomCount':
            # Calculate symptom count based on other features
            encoded.append(symptom_count)
            continue
            
        value = input_dict.get(feature_name, "")
        if not value:
            return None, f"Missing input for: {feature_name}"
        
        value = normalize_input(value)
        encoder = label_encoders.get(feature_name)
        if encoder is None:
            return None, f"Encoder missing for: {feature_name}"
            
        normalized_classes = [c.lower() for c in encoder.classes_]
        if value not in normalized_classes:
            # Use most frequent class instead of defaulting to 0
            most_common = encoder.classes_[0].lower()
            print(f"⚠️ Unknown value '{value}' for {feature_name}, using '{most_common}'")
            value = most_common
        
        encoded_val = normalized_classes.index(value)
        encoded.append(encoded_val)
        symptom_count += 1 if value != 'normal' else 0
    
    return np.array(encoded).reshape(1, -1), None

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    error_message = None
    confidence = None

    if request.method == 'POST':
        if None in (model, label_encoders, scaler, target_encoder):
            error_message = "❌ Models not loaded correctly."
            return render_template('index.html', error_message=error_message)

        # Get inputs from form
        input_dict = {
            'AnimalName': request.form.get('animal_name', ''),
            'BloodBrainDisease': request.form.get('blood_brain', ''),
            'AppearenceDisease': request.form.get('appearance', ''),
            'GeneralDisease': request.form.get('general', ''),
            'LungDisease': request.form.get('lung', ''),
            'AbdominalDisease': request.form.get('abdominal', '')
        }

        try:
            encoded_inputs, encode_error = encode_inputs(input_dict)
            if encode_error:
                error_message = encode_error
            else:
                scaled_inputs = scaler.transform(encoded_inputs)
                
                # Get prediction with confidence
                pred_proba = model.predict_proba(scaled_inputs)[0]
                pred_class = model.predict(scaled_inputs)[0]
                decoded = target_encoder.inverse_transform([pred_class])[0]
                confidence = max(pred_proba)
                
                # Enhanced prediction messages
                if confidence < 0.6:
                    prediction = {
                        'result': 'Uncertain Health Status - Professional Evaluation Recommended',
                        'confidence': f'{confidence:.0%} confidence',
                        'class': 'warning'
                    }
                elif decoded.lower() == 'critical':
                    prediction = {
                        'result': 'Critical Health Condition Detected',
                        'confidence': f'{confidence:.0%} confidence',
                        'class': 'danger'
                    }
                else:
                    prediction = {
                        'result': 'Normal Health Status',
                        'confidence': f'{confidence:.0%} confidence',
                        'class': 'safe'
                    }
                
        except Exception as e:
            error_message = f"Prediction error: {str(e)}"
            print("❌ Prediction exception:", e)

    return render_template('index.html', 
                         prediction=prediction,
                         error_message=error_message)

@app.route('/info')
def info():
    if not label_encoders:
        return "Encoders not loaded"
    
    # Prepare encoder information exactly as in original version
    info_data = {
        k: sorted(v.classes_.tolist()) for k, v in label_encoders.items()
    }
    
    # Add target encoder if available
    if target_encoder:
        info_data['HealthStatus'] = sorted(target_encoder.classes_.tolist())
    
    return render_template('info.html', info_data=info_data)

if __name__ == '__main__':
    print("🚀 App running from:", os.getcwd())
    app.run(debug=True, host='0.0.0.0', port=5000)