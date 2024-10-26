import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Load the test dataset
data = pd.read_csv('test_dataset.csv')

# Select the same features used in training
features = [
    'Day_of_Week', 'Junction_Control', 'Junction_Detail', 'Latitude', 'Longitude',
    'Light_Conditions', 'Local_Authority_(District)', 'Carriageway_Hazards', 
    'Number_of_Casualties', 'Number_of_Vehicles', 'Police_Force', 
    'Road_Surface_Conditions', 'Road_Type', 'Speed_limit', 'Time', 
    'Urban_or_Rural_Area', 'Weather_Conditions', 'Vehicle_Type'
]
X_test = data[features]
y_test = data['Accident_Severity']

# Encode the test data and map target labels
X_test = pd.get_dummies(X_test, drop_first=True)
y_test = y_test.map({'Slight': 0, 'Serious': 1, 'Fatal': 2, 'Fetal': 3})

# Ensure the test dataset matches the training feature set
X_test = X_test.reindex(columns=model.get_params()['feature_names_in_'], fill_value=0)

# Load the saved model
model = joblib.load('accident_severity_model.pkl')

# Make predictions on the test dataset
y_pred = model.predict(X_test)

# Generate classification report and confusion matrix
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Slight', 'Serious', 'Fatal', 'Fetal']))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
