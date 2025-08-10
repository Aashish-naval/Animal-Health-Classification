# Animal Health Classification Web App

A Flask-based web application for classifying animal health status based on symptoms and animal type.

## Folder Structure

```
animal_health_webapp/
│
├── app.py                     # Main Flask application
├── requirements.txt           # Python dependencies
├── README.md                 # This file
│
├── models/                   # Model files directory
│   ├── best_model.pkl        # Trained classifier
│   ├── label_encoders.pkl    # Feature encoders
│   └── scaler.pkl           # Data scaler
│
└── templates/               # HTML templates
    ├── index.html          # Main form page
    └── info.html           # Available options reference
```

## Setup Instructions

1. **Create the project directory:**
   ```bash
   mkdir animal_health_webapp
   cd animal_health_webapp
   ```

2. **Copy your trained model files:**
   ```bash
   mkdir models
   # Copy your three .pkl files to the models/ directory:
   # - best_model.pkl
   # - label_encoders.pkl  
   # - scaler.pkl
   ```

3. **Create templates directory:**
   ```bash
   mkdir templates
   # Add index.html and info.html to this directory
   ```

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Run the application:**
   ```bash
   python app.py
   ```

6. **Access the application:**
   - Open your browser and go to `http://localhost:5000`
   - Fill in the form with animal name and 5 symptoms
   - Click "Predict Health Status" to get results

## Features

- **Input Validation**: Checks if entered values exist in the training data
- **Automatic Normalization**: Inputs are automatically formatted (capitalized)
- **Error Handling**: Clear error messages for invalid inputs
- **Responsive Design**: Works on desktop and mobile devices
- **Reference Guide**: `/info` page shows all valid input options
- **Real-time Feedback**: Visual indicators for form validation

## Usage Examples

### Valid Inputs (based on your training data):
- **Animal Name**: Dog, Cat, Cow, Buffaloes, etc.
- **Symptoms**: Fever, Diarrhea, Coughing, Vomiting, Weight loss, Lethargy, etc.

### Input Format:
- Inputs are case-insensitive
- Leading/trailing spaces are automatically removed
- First letter is automatically capitalized

## API Endpoints

- `GET /` - Main form page
- `POST /` - Submit prediction request
- `GET /info` - View available input options

## Error Messages

The app provides specific error messages for:
- Empty fields
- Unknown animal names
- Unknown symptoms
- Model loading issues

## Technical Details

- **Backend**: Flask (Python)
- **ML Model**: Scikit-learn classifier with label encoding and scaling
- **Frontend**: HTML5, CSS3, JavaScript
- **Data Processing**: Automatic normalization and validation